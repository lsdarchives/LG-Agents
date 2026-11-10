"""Local FastAPI server for streaming DevTeam runs into the office viewer."""

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from devteam.team_graph import (
    build_pipeline,
    new_run_id,
    open_checkpointer,
    plan_response_command,
    run_config,
)
from devteam.team_state import initial_team_state

OFFICE_ROOT = Path(__file__).resolve().parent / "office"
app = FastAPI(title="DevTeam Office", version="1.0")
sessions: dict[str, dict] = {}


class NewRun(BaseModel):
    request: str = Field(min_length=1, max_length=10_000)


def public_state(state: dict) -> dict:
    """Return only the run details needed by the office client."""
    return {
        "run_id": state.get("run_id"),
        "request": state.get("request", ""),
        "status": state.get("status", "unknown"),
        "reason": state.get("reason", ""),
        "task_board": state.get("task_board", []),
        "artifacts": state.get("artifacts", {}),
        "activations": state.get("activations", 0),
    }


def run_stream(session: dict, loop: asyncio.AbstractEventLoop, action: str, request: str = "", task_board=None) -> dict:
    """Run or resume a graph in a worker thread, forwarding custom events to its WebSocket."""
    run_id = session["run_id"]

    def publish(payload: dict) -> None:
        loop.call_soon_threadsafe(_publish_on_loop, session, payload)

    if action == "start":
        graph = build_pipeline(open_checkpointer())
        config = run_config(run_id)
        graph_input = initial_team_state(request, run_id)
    else:
        graph, config, graph_input = plan_response_command(run_id, action, task_board)

    for chunk in graph.stream(graph_input, config, stream_mode="custom"):
        if isinstance(chunk, dict) and "event" in chunk:
            publish({"kind": "event", "event": chunk["event"]})

    snapshot = graph.get_state(config)
    return public_state(snapshot.values)


def _publish_on_loop(session: dict, payload: dict) -> None:
    session["history"].append(payload)
    queue = session.get("queue")
    if queue is not None:
        queue.put_nowait(payload)


async def run_in_background(session: dict, action: str, request: str = "", task_board=None) -> None:
    session["active"] = True
    session["status"] = "running"
    loop = asyncio.get_running_loop()
    try:
        state = await asyncio.to_thread(run_stream, session, loop, action, request, task_board)
        session["status"] = state["status"]
        _publish_on_loop(session, {"kind": "state", "state": state})
    except (Exception, SystemExit) as exc:  # report graph and approval errors to the connected UI
        _publish_on_loop(session, {"kind": "error", "message": str(exc)})
        session["status"] = "awaiting_approval" if action in {"approve", "reject"} else "failed"
    finally:
        session["active"] = False


@app.post("/api/runs")
async def create_run(body: NewRun) -> dict:
    run_id = new_run_id()
    sessions[run_id] = {
        "run_id": run_id,
        "history": [],
        "queue": None,
        "active": False,
        "status": "created",
    }
    return {"run_id": run_id}


@app.websocket("/ws/runs/{run_id}")
async def run_socket(websocket: WebSocket, run_id: str) -> None:
    session = sessions.get(run_id)
    if session is None:
        await websocket.close(code=4404, reason="Unknown run id")
        return

    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    session["queue"] = queue
    history = list(session["history"])
    try:
        for message in history:
            await websocket.send_json(message)
        await websocket.send_json({"kind": "connected", "run_id": run_id})

        async def read_commands() -> None:
            while True:
                try:
                    command = await websocket.receive_json()
                except WebSocketDisconnect:
                    queue.put_nowait(None)
                    return
                except ValueError as exc:
                    _publish_on_loop(session, {"kind": "error", "message": f"Invalid WebSocket JSON: {exc}"})
                    continue
                action = command.get("action") if isinstance(command, dict) else None
                if action == "start":
                    request = command.get("request")
                    if session["active"] or session["status"] != "created":
                        _publish_on_loop(session, {"kind": "error", "message": "This run has already started."})
                    elif not isinstance(request, str) or not request.strip():
                        _publish_on_loop(session, {"kind": "error", "message": "Enter a project request first."})
                    else:
                        session["active"] = True
                        asyncio.create_task(run_in_background(session, "start", request.strip()))
                elif action in {"approve", "reject"}:
                    if session["active"] or session["status"] != "awaiting_approval":
                        _publish_on_loop(session, {"kind": "error", "message": "This run is not waiting for approval."})
                    else:
                        board = command.get("task_board") if action == "approve" else None
                        session["active"] = True
                        asyncio.create_task(run_in_background(session, action, task_board=board))
                else:
                    _publish_on_loop(session, {"kind": "error", "message": "Unknown office command."})

        reader = asyncio.create_task(read_commands())
        try:
            while True:
                message = await queue.get()
                if message is None:
                    break
                await websocket.send_json(message)
        finally:
            reader.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        if session.get("queue") is queue:
            session["queue"] = None


app.mount("/", StaticFiles(directory=OFFICE_ROOT, html=True), name="office")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
