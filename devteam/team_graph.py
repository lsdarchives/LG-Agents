"""v2 team graph with a SQLite checkpointer, so any run can be resumed by its id.

Step 13 wires a FIXED planning pipeline (product_manager -> architect -> project_manager); step 14
replaces the fixed edges with the Tech Lead supervisor. (Named team_graph.py: v1 owns graph.py.)

    python -m devteam.team_graph "build a ..."          # new run
    python -m devteam.team_graph --resume <run_id>      # continue a killed/crashed run
"""

import argparse
import sqlite3
import sys
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from devteam.artifacts import load_inputs, sync_state
from devteam.builder import build_agent
from devteam.config import CHECKPOINT_DB, RECURSION_LIMIT, RUNS_ROOT, WORKSPACE_ROOT
from devteam.events import EventLog
from devteam.registry import load_registry
from devteam.team_state import TeamState, initial_team_state
from devteam.toolbox import Workspace

PIPELINE = ["product_manager", "architect", "project_manager"]


def new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def workspace_for(run_id: str) -> Workspace:
    return Workspace(WORKSPACE_ROOT / run_id)


def events_for(run_id: str) -> EventLog:
    return EventLog(RUNS_ROOT / run_id / "events.jsonl")


def agent_node(name: str):
    """A graph node that runs one registry agent on the current state."""
    registry = load_registry()
    spec = registry.get(name)
    run_agent = build_agent(spec, registry.shared_rules)

    def node(state: TeamState) -> dict:
        ws = workspace_for(state["run_id"])
        out = run_agent(load_inputs(spec.reads, state, ws), ws, events_for(state["run_id"]))
        result = out["result"].model_dump()
        return {
            **sync_state(ws),
            "last_result": {"agent": name, **result},
            "activations": 1,
            "messages": [AIMessage(content=f"{result['status']}: {result['summary']}", name=name)],
        }

    node.__name__ = name
    return node


def finish(state: TeamState) -> dict:
    status = "done" if state["task_board"] else "partial"
    reason = f"{len(state['task_board'])} tasks planned" if state["task_board"] else "no task board produced"
    events_for(state["run_id"]).emit("finish", status=status, reason=reason)
    print(f"[finish] status={status} reason={reason}")
    return {"status": status, "reason": reason}


def build_pipeline(checkpointer=None):
    builder = StateGraph(TeamState)
    previous = START
    for name in PIPELINE:
        builder.add_node(name, agent_node(name))
        builder.add_edge(previous, name)
        previous = name
    builder.add_node("finish", finish)
    builder.add_edge(previous, "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)


def open_checkpointer() -> SqliteSaver:
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False))


def run_config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}


def start(request: str, run_id: str | None = None) -> dict:
    run_id = run_id or new_run_id()
    print(f"[run] id={run_id} workspace={WORKSPACE_ROOT / run_id}")
    graph = build_pipeline(open_checkpointer())
    return graph.invoke(initial_team_state(request, run_id), run_config(run_id))


def resume(run_id: str) -> dict:
    """Continue a run from its last checkpoint (the node that was running when it died reruns)."""
    graph = build_pipeline(open_checkpointer())
    config = run_config(run_id)
    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise SystemExit(f"no checkpoint for run {run_id!r}")
    if not snapshot.next:
        print(f"[resume] run {run_id} already finished (status={snapshot.values['status']})")
        return snapshot.values
    print(f"[resume] run {run_id}: continuing at {list(snapshot.next)}")
    return graph.invoke(None, config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run (or resume) the v2 planning pipeline.")
    parser.add_argument("request", nargs="?", help="what to build")
    parser.add_argument("--run-id", help="id for a new run (default: timestamp)")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue a run from its last checkpoint")
    args = parser.parse_args()
    if args.resume:
        state = resume(args.resume)
    elif args.request:
        state = start(args.request, args.run_id)
    else:
        parser.error("give a request or --resume RUN_ID")
    print(f"[run] status={state['status']} activations={state['activations']} "
          f"artifacts={sorted(state['artifacts'])} tasks={len(state['task_board'])}")
    return 0 if state["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
