"""Step 16 graph check with deterministic agents; never calls a provider or runs project code."""

import json

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from devteam import team_graph
from devteam.builder import AgentResult
from devteam.events import read_events
from devteam.team_state import initial_team_state


def test_approved_plan_runs_test_build_qa_per_dependency_ready_task(tmp_path, monkeypatch):
    monkeypatch.setattr(team_graph, "WORKSPACE_ROOT", tmp_path / "workspace")
    monkeypatch.setattr(team_graph, "RUNS_ROOT", tmp_path / "runs")
    quality_tasks = []

    def fake_build_agent(spec, shared_rules, handoff_targets=None, task_id=None):
        def fake_agent(inputs, workspace, events, objective=None):
            events.emit("activate", agent=spec.name, objective=objective or spec.description, reads=list(inputs))
            handoff = None
            if spec.name == "tech_lead":
                target = next(iter(handoff_targets))
                handoff = {
                    "target": target,
                    "objective": f"Create {target} planning artifact",
                    "inputs": "existing planning artifacts",
                    "output": f"Save the {target} artifact",
                    "boundaries": "Do not implement code",
                }
                events.emit(
                    "handoff", **{"from": "tech_lead"}, to=target, objective=handoff["objective"]
                )
                summary = f"Routed to {target}"
            elif spec.name in {"product_manager", "architect", "project_manager"}:
                artifact = {"product_manager": "prd", "architect": "design", "project_manager": "tasks.json"}[spec.name]
                content = "# Planning\n"
                if artifact == "tasks.json":
                    content = json.dumps(
                        [
                            {
                                "id": "T1",
                                "title": "Create URL storage",
                                "owner_role": "backend_dev",
                                "stack": "python-fastapi",
                                "files": ["app.py"],
                                "depends_on": [],
                                "acceptance": ["AC1"],
                                "status": "todo",
                            },
                            {
                                "id": "T2",
                                "title": "Add redirect endpoint",
                                "owner_role": "backend_dev",
                                "stack": "python-fastapi",
                                "files": ["app.py"],
                                "depends_on": ["T1"],
                                "acceptance": ["AC2"],
                                "status": "todo",
                            },
                        ]
                    )
                workspace.write_artifact(artifact, content)
                summary = f"Saved {artifact}"
            elif spec.name == "test_engineer":
                task = json.loads(inputs["task"])
                workspace.create_file(task["test_file"], "def test_contract_placeholder():\n    assert True\n")
                summary = f"Wrote tests for {task['id']}"
            elif spec.name == "backend_dev":
                task = json.loads(inputs["task"])
                summary = f"Implemented {task['id']}"
            elif spec.name == "debugger":
                summary = "No diagnosis needed in the passing scenario"
            else:
                raise AssertionError(f"Unexpected LLM role: {spec.name}")

            result = AgentResult(status="done", summary=summary)
            events.emit("result", agent=spec.name, status="done", summary=summary)
            return {"result": result, "raw": "", "handoff": handoff}

        return fake_agent

    def fake_quality(workspace, task_id, test_file=None):
        quality_tasks.append((task_id, test_file))
        if task_id == "T1" and quality_tasks.count(("T1", test_file)) == 1:
            return {
                "task_id": task_id,
                "stack": "python",
                "passed": 0,
                "failed": 1,
                "lint_issues": 1,
                "tests_clean": False,
                "lint_clean": False,
                "output": "1 failed; ruff found one issue",
            }
        return {
            "task_id": task_id,
            "stack": "python",
            "passed": 1,
            "failed": 0,
            "lint_issues": 0,
            "tests_clean": True,
            "lint_clean": True,
            "output": "1 passed; ruff clean",
        }

    monkeypatch.setattr(team_graph, "build_agent", fake_build_agent)
    monkeypatch.setattr(team_graph, "run_task_checks", fake_quality)
    run_id = "offline-task-loop"
    graph = team_graph.build_pipeline(MemorySaver())
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}
    graph.invoke(initial_team_state("build a small URL service", run_id), config)
    state = graph.invoke(Command(resume={"action": "approve"}), config)
    events = read_events(tmp_path / "runs" / run_id / "events.jsonl")

    assert state["status"] == "done"
    assert [task["status"] for task in state["task_board"]] == ["done", "done"]
    assert [task_id for task_id, _ in quality_tasks] == ["T1", "T2"]
    assert [test_file for _, test_file in quality_tasks] == ["tests/test_t1.py", "tests/test_t2.py"]
    assert [event["to"] for event in events if event["type"] == "handoff"] == [
        "product_manager",
        "architect",
        "project_manager",
        "approval",
        "test_engineer",
        "backend_dev",
        "qa_runner",
        "debugger",
        "backend_dev",
        "qa_runner",
        "tech_lead",
        "test_engineer",
        "backend_dev",
        "qa_runner",
        "tech_lead",
    ]
    assert sum(event["agent"] == "debugger" for event in events if event["type"] == "result") == 1
    assert [event["failed"] for event in events if event["type"] == "tests"] == [1, 0, 0]
