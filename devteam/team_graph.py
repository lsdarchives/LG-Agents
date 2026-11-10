"""v2 team graph with a SQLite checkpointer, so any run can be resumed by its id.

Step 13 wires a FIXED planning pipeline (product_manager -> architect -> project_manager); step 14
replaces the fixed edges with the Tech Lead supervisor. (Named team_graph.py: v1 owns graph.py.)

    python -m devteam.team_graph "build a ..."          # new run
    python -m devteam.team_graph --resume <run_id>      # continue a killed/crashed run
"""

import argparse
import json
import sqlite3
import sys
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from devteam.artifacts import load_inputs, sync_state
from devteam.builder import build_agent
from devteam.config import (
    CHECKPOINT_DB,
    MAX_ACTIVATIONS,
    RECURSION_LIMIT,
    RUNS_ROOT,
    WORKSPACE_ROOT,
)
from devteam.events import EventLog
from devteam.registry import load_registry
from devteam.team_state import TeamState, initial_team_state
from devteam.toolbox import Workspace

PLANNING_AGENTS = ["product_manager", "architect", "project_manager"]


def next_planning_targets(artifacts: dict[str, str], last_result: dict | None = None) -> dict[str, list[str]]:
    """Return only the role that owns the next missing planning artifact."""
    artifact_to_agent = {
        "prd": "product_manager",
        "design": "architect",
        "tasks": "project_manager",
    }
    target = next((agent for artifact, agent in artifact_to_agent.items() if artifact not in artifacts), None)
    if target is None:
        return {"FINISH": []}
    if last_result and last_result.get("agent") == target and last_result.get("status") != "done":
        return {"FINISH": []}
    registry = load_registry()
    return {target: [read for read in registry.get(target).reads if read != "handoff"]}


def planning_terminal(state: TeamState, artifacts: dict[str, str]) -> tuple[str, str] | None:
    """Return a terminal planning status when the last worker completed or could not proceed."""
    result = state.get("last_result") or {}
    if result.get("agent") in PLANNING_AGENTS and result.get("status") in {"blocked", "failed"}:
        return "partial", f"{result['agent']} {result['status']}: {result.get('summary', 'planning stopped')}"
    if (
        "tasks" in artifacts
        and state["task_board"]
        and result.get("agent") == "project_manager"
        and result.get("status") == "done"
    ):
        return "done", f"Planning complete: {len(state['task_board'])} tasks planned"
    return None


def new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def workspace_for(run_id: str) -> Workspace:
    return Workspace(WORKSPACE_ROOT / run_id)


def events_for(run_id: str) -> EventLog:
    return EventLog(RUNS_ROOT / run_id / "events.jsonl")


def agent_node(name: str):
    """Build a worker node or the Tech Lead node that dynamically routes by Command."""
    registry = load_registry()
    spec = registry.get(name)
    run_agent = None if name == "tech_lead" else build_agent(spec, registry.shared_rules)

    def node(state: TeamState) -> dict | Command:
        if state["activations"] >= MAX_ACTIVATIONS:
            return Command(
                goto="finish",
                update={"status": "partial", "reason": f"maximum of {MAX_ACTIVATIONS} agent activations reached"},
            )
        ws = workspace_for(state["run_id"])
        current_agent = run_agent
        if name == "tech_lead":
            terminal = planning_terminal(state, ws.artifacts())
            if terminal:
                status, reason = terminal
                log = events_for(state["run_id"])
                log.emit("activate", agent="tech_lead", objective=reason, reads=list(spec.reads))
                log.emit("handoff", **{"from": "tech_lead"}, to="finish", objective=reason)
                log.emit("result", agent="tech_lead", status="done", summary=reason)
                return Command(
                    goto="finish",
                    update={
                        **sync_state(ws),
                        "last_result": {"agent": "tech_lead", "status": "done", "summary": reason},
                        "activations": 1,
                        "messages": [AIMessage(content=f"done: {reason}", name="tech_lead")],
                        "handoff": None,
                        **({"status": "partial", "reason": reason} if status == "partial" else {}),
                    },
                )
            targets = next_planning_targets(ws.artifacts(), state.get("last_result"))
            current_agent = build_agent(spec, registry.shared_rules, handoff_targets=targets)
        extra = None
        if name in PLANNING_AGENTS and state.get("handoff"):
            extra = {"handoff": json.dumps(state["handoff"], indent=2)}
        objective = (state.get("handoff") or {}).get("objective")
        out = current_agent(
            load_inputs(spec.reads, state, ws, extra=extra),
            ws,
            events_for(state["run_id"]),
            objective=objective,
        )
        result = out["result"].model_dump()
        update = {
            **sync_state(ws),
            "last_result": {"agent": name, **result},
            "activations": 1,
            "messages": [AIMessage(content=f"{result['status']}: {result['summary']}", name=name)],
        }
        if name == "tech_lead":
            decision = out["handoff"]
            if not decision:
                previous = state.get("last_result") or {}
                reason = "Tech Lead returned without an accepted handoff."
                if previous.get("status") in {"blocked", "failed"}:
                    reason = (
                        f"{previous.get('agent', 'An agent')} {previous['status']}: "
                        f"{previous.get('summary', 'planning could not continue')}"
                    )
                events_for(state["run_id"]).emit(
                    "handoff",
                    **{"from": "tech_lead"},
                    to="finish",
                    objective=f"Stop planning: {reason}",
                )
                return Command(
                    goto="finish",
                    update={
                        **update,
                        "status": "partial",
                        "reason": reason,
                        "handoff": None,
                    },
                )
            if decision["target"] == "FINISH":
                previous = state.get("last_result") or {}
                if previous.get("status") in {"blocked", "failed"}:
                    update.update(
                        status="partial",
                        reason=(
                            f"{previous.get('agent', 'An agent')} {previous['status']}: "
                            f"{previous.get('summary', 'planning could not continue')}"
                        ),
                    )
                elif "tasks" not in ws.artifacts():
                    update.update(
                        status="partial",
                        reason="Tech Lead ended planning before tasks.json was produced.",
                    )
                return Command(goto="finish", update={**update, "handoff": None})
            return Command(goto=decision["target"], update={**update, "handoff": decision})
        return {**update, "handoff": None}

    node.__name__ = name
    return node


def finish(state: TeamState) -> dict:
    if state["status"] == "partial":
        status = "partial"
        reason = state["reason"] or "run ended before planning was complete"
    else:
        status = "done" if state["task_board"] else "partial"
        reason = f"{len(state['task_board'])} tasks planned" if state["task_board"] else "no task board produced"
    events_for(state["run_id"]).emit("finish", status=status, reason=reason)
    print(f"[finish] status={status} reason={reason}")
    return {"status": status, "reason": reason}


def build_pipeline(checkpointer=None):
    builder = StateGraph(TeamState)
    builder.add_node("tech_lead", agent_node("tech_lead"))
    for name in PLANNING_AGENTS:
        builder.add_node(name, agent_node(name))
        builder.add_edge(name, "tech_lead")
    builder.add_node("finish", finish)
    builder.add_edge(START, "tech_lead")
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
