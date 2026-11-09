"""Step 13 offline checks: reducers, event validation, artifact loading."""

import json

import pytest

from devteam.artifacts import artifact_index, load_inputs, sync_state
from devteam.events import EventLog, read_events
from devteam.team_state import initial_team_state, merge_dicts, merge_tasks
from devteam.toolbox import Workspace


def test_merge_tasks_by_id():
    old = [{"id": "T1", "title": "a", "status": "todo"}, {"id": "T2", "title": "b", "status": "todo"}]
    new = [{"id": "T2", "status": "done"}, {"id": "T3", "title": "c", "status": "todo"}]
    merged = merge_tasks(old, new)
    assert [t["id"] for t in merged] == ["T1", "T2", "T3"]
    assert merged[1] == {"id": "T2", "title": "b", "status": "done"}


def test_merge_dicts_keeps_old_keys():
    assert merge_dicts({"prd": "a"}, {"design": "b"}) == {"prd": "a", "design": "b"}


def test_events_validate_and_reject(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.emit("activate", agent="architect", objective="design it", reads=["prd"])
    log.emit("handoff", **{"from": "tech_lead"}, to="architect", objective="design it")
    log.emit("finish", status="done")
    events = read_events(tmp_path / "events.jsonl")
    assert [e["type"] for e in events] == ["activate", "handoff", "finish"]
    assert events[1]["from"] == "tech_lead"
    with pytest.raises(ValueError):
        log.emit("result", agent="x", status="maybe", summary="s")  # bad status
    with pytest.raises(ValueError):
        log.emit("teleport", agent="x")  # unknown type


def test_resumed_log_keeps_one_timeline(tmp_path):
    path = tmp_path / "events.jsonl"
    first = EventLog(path).emit("finish", status="partial")
    again = EventLog(path)  # a resumed run reopens the same file
    assert again.start == first["ts"]


def test_load_inputs_and_sync(tmp_path):
    ws = Workspace(tmp_path / "run")
    ws.write_artifact("prd", "# PRD\nGoal")
    ws.write_artifact("tasks.json", json.dumps([{"id": "T1", "title": "x", "status": "todo"}]))
    state = initial_team_state("build x", "r1")
    inputs = load_inputs(["request", "prd", "design", "artifact_index", "task_board"], state, ws)
    assert inputs["request"] == "build x" and inputs["prd"] == "# PRD\nGoal"
    assert "design" not in inputs  # missing -> builder shows "(not provided)"
    assert "task_board" in inputs  # falls back to tasks.json when state has no board yet
    update = sync_state(ws)
    assert update["task_board"][0]["id"] == "T1"
    assert "prd (prd.md, 2 lines): PRD" in artifact_index(ws)


def test_stack_catalog_lists_python_profiles():
    from devteam.artifacts import stack_catalog

    catalog = stack_catalog()
    assert catalog.startswith("default: python-fastapi")
    assert "- python-fastapi (python, backend):" in catalog
