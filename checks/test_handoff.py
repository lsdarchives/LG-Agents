"""Step 14 offline checks: routing decisions are validated and the graph exposes its destinations."""

import json

from devteam.team_graph import (
    PLANNING_AGENTS,
    build_pipeline,
    next_planning_targets,
    planning_terminal,
)
from devteam.team_state import initial_team_state
from devteam.toolbox import Workspace, make_tools


def test_handoff_accepts_only_planning_targets_and_complete_delegations(tmp_path):
    ws = Workspace(tmp_path / "run")
    inputs_by_agent = {
        "product_manager": ["request", "domain_notes"],
        "architect": ["prd", "domain_notes", "stack_catalog"],
        "project_manager": ["prd", "design"],
    }
    tools, missing = make_tools(ws, ["handoff"], inputs_by_agent)
    assert missing == []
    handoff = tools[0]

    accepted = json.loads(
        handoff.invoke(
            {
                "target": "architect",
                "objective": "Design the URL shortener from the PRD.",
                "output": "Save design.md.",
                "boundaries": "Do not write implementation code.",
            }
        )
    )
    assert accepted["accepted"] is True
    assert accepted["target"] == "architect"
    assert accepted["inputs"] == "prd, domain_notes, stack_catalog"

    invalid_target = handoff.invoke({"target": "backend_dev", "objective": "Build it"})
    assert invalid_target.startswith("ERROR:") and "target must be one of" in invalid_target
    incomplete = handoff.invoke({"target": "architect", "objective": "Design it", "inputs": "prd"})
    assert incomplete.startswith("ERROR:") and "require output and boundaries" in incomplete


def test_handoff_can_finish(tmp_path):
    ws = Workspace(tmp_path / "run")
    tools, _ = make_tools(ws, ["handoff"], PLANNING_AGENTS)
    decision = json.loads(tools[0].invoke({"target": "FINISH", "objective": "Planning is complete."}))
    assert decision["target"] == "FINISH"


def test_supervisor_graph_has_only_phase_14_destinations():
    graph = build_pipeline()
    assert set(PLANNING_AGENTS + ["tech_lead", "finish"]) <= set(graph.get_graph().nodes)


def test_supervisor_only_exposes_the_owner_of_the_next_missing_artifact():
    assert list(next_planning_targets({})) == ["product_manager"]
    assert list(next_planning_targets({"prd": ".devteam/prd.md"})) == ["architect"]
    assert list(next_planning_targets({"prd": "p", "design": "d"})) == ["project_manager"]
    assert next_planning_targets({"prd": "p", "design": "d", "tasks": "t"}) == {"FINISH": []}
    assert next_planning_targets({}, {"agent": "product_manager", "status": "failed"}) == {"FINISH": []}


def test_planning_finishes_when_complete_and_stops_partial_when_blocked():
    state = initial_team_state("build it", "r1")
    state["task_board"] = [{"id": "T1", "title": "Build it"}]
    state["last_result"] = {"agent": "project_manager", "status": "done"}
    assert planning_terminal(state, {"prd": "p", "design": "d", "tasks": "t"}) == (
        "done",
        "Planning complete: 1 tasks planned",
    )

    state["last_result"] = {"agent": "project_manager", "status": "blocked", "summary": "needs IDs"}
    assert planning_terminal(state, {"prd": "p", "design": "d", "tasks": "t"}) == (
        "partial",
        "project_manager blocked: needs IDs",
    )
