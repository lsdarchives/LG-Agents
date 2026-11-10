"""Step 14 graph check with deterministic agents; exercises Command routes without API calls."""

from langgraph.checkpoint.memory import MemorySaver

from devteam import team_graph
from devteam.builder import AgentResult
from devteam.events import read_events
from devteam.team_state import initial_team_state


def test_supervisor_routes_all_planners_and_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(team_graph, "WORKSPACE_ROOT", tmp_path / "workspace")
    monkeypatch.setattr(team_graph, "RUNS_ROOT", tmp_path / "runs")

    def fake_build_agent(spec, shared_rules, handoff_targets=None):
        def fake_agent(inputs, workspace, events, objective=None):
            events.emit(
                "activate",
                agent=spec.name,
                objective=objective or spec.description,
                reads=list(inputs),
            )
            if spec.name == "tech_lead":
                target = next(iter(handoff_targets))
                decision = {
                    "target": target,
                    "objective": f"Create {target} planning artifact",
                    "inputs": "existing planning artifacts",
                    "output": f"Save the {target} artifact",
                    "boundaries": "Do not implement code",
                }
                events.emit(
                    "handoff",
                    **{"from": "tech_lead"},
                    to="finish" if target == "FINISH" else target,
                    objective=decision["objective"],
                )
                summary = f"Routed to {target}"
                status = "done"
                handoff = decision
            else:
                name = {"product_manager": "prd", "architect": "design", "project_manager": "tasks.json"}[
                    spec.name
                ]
                content = '[{"id":"T1","title":"Build feature"}]' if name == "tasks.json" else f"# {name}\n"
                workspace.write_artifact(name, content)
                events.emit("artifact_written", agent=spec.name, name=name)
                summary = f"Saved {name}"
                status = "done"
                handoff = None
            result = AgentResult(status=status, summary=summary)
            events.emit("result", agent=spec.name, status=status, summary=summary)
            return {"result": result, "handoff": handoff}

        return fake_agent

    monkeypatch.setattr(team_graph, "build_agent", fake_build_agent)
    run_id = "offline-phase03"
    graph = team_graph.build_pipeline(MemorySaver())
    state = graph.invoke(
        initial_team_state("build a small API", run_id),
        {"configurable": {"thread_id": run_id}, "recursion_limit": 200},
    )
    events = read_events(tmp_path / "runs" / run_id / "events.jsonl")
    routes = [event["to"] for event in events if event["type"] == "handoff"]

    assert routes == ["product_manager", "architect", "project_manager", "finish"]
    assert state["status"] == "done"
    assert len(state["task_board"]) == 1
    assert {"prd", "design", "tasks"} <= set(state["artifacts"])
