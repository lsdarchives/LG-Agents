"""Step 12 offline checks: workspace tools stay inside the workspace and refuse bad edits."""

import pytest

from devteam.toolbox import ToolError, Workspace, make_tools
from devteam.toolbox.workspace import VIEW_LINES


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "run")


@pytest.mark.parametrize("path", ["../outside.txt", r"..\..\x.py", "/etc/passwd", "C:/Windows/x.txt", "a/../../b.txt"])
def test_path_escape_rejected(ws, path):
    with pytest.raises(ToolError, match="outside the workspace"):
        ws.create_file(path, "x")


def test_artifact_dir_is_off_limits_to_file_tools(ws):
    with pytest.raises(ToolError, match="read_artifact"):
        ws.create_file(".devteam/prd.md", "x")


def test_bad_python_edit_rejected_file_unchanged(ws):
    ws.create_file("app/main.py", "def f():\n    return 1\n")
    before = (ws.root / "app/main.py").read_text()
    with pytest.raises(ToolError, match="would not parse"):
        ws.edit_file("app/main.py", 2, 2, "    return (")
    assert (ws.root / "app/main.py").read_text() == before


def test_bad_python_create_rejected(ws):
    with pytest.raises(ToolError, match="would not parse"):
        ws.create_file("bad.py", "def (:\n")
    assert not (ws.root / "bad.py").exists()


def test_create_refuses_overwrite(ws):
    ws.create_file("a.txt", "one")
    with pytest.raises(ToolError, match="already exists"):
        ws.create_file("a.txt", "two")


def test_edit_replaces_and_inserts(ws):
    ws.create_file("m.py", "a = 1\nb = 2\nc = 3\n")
    out = ws.edit_file("m.py", 2, 2, "b = 20")
    assert "b = 20" in out
    ws.edit_file("m.py", 1, 0, "import os")  # insert before line 1
    assert (ws.root / "m.py").read_text() == "import os\na = 1\nb = 20\nc = 3\n"


def test_viewer_pages(ws):
    ws.create_file("long.txt", "\n".join(f"line {i}" for i in range(1, 251)))
    first = ws.view_file("long.txt")
    assert first.startswith(f"[long.txt: lines 1-{VIEW_LINES} of 250]")
    assert "start=101" in first
    last = ws.view_file("long.txt", start=201)
    assert "lines 201-250 of 250" in last and " 250 | line 250" in last and "more lines" not in last


def test_search_returns_file_list_only(ws):
    ws.create_file("a.py", "token = 1\nprint(token)\n")
    ws.create_file("b.py", "x = 2\n")
    out = ws.search_files("TOKEN")
    assert out.splitlines() == ["1 files contain 'TOKEN':", "a.py (2 matches)"]


def test_tree_skips_caches_and_artifacts(ws):
    ws.create_file("src/app.py", "x = 1\n")
    (ws.root / "__pycache__").mkdir()
    ws.write_artifact("prd", "# PRD")
    assert ws.list_tree() == "src/\n  app.py"


def test_artifacts_roundtrip(ws):
    assert ws.write_artifact("prd", "# PRD\nGoal").startswith("Saved artifact prd.md")
    assert ws.read_artifact("prd") == "# PRD\nGoal"
    ws.write_artifact("tasks.json", "[]")
    assert ws.artifacts() == {"prd": ".devteam/prd.md", "tasks": ".devteam/tasks.json"}
    with pytest.raises(ToolError, match="no artifact"):
        ws.read_artifact("design")


def test_task_board_alias_is_saved_as_json(ws):
    content = '[{"id":"T1","title":"A task"}]'
    ws.write_artifact("task_board", content)
    ws.write_artifact("task_board.md", content)
    assert ws.read_artifact("tasks.json") == ws.read_artifact("task_board")
    assert ws.artifacts() == {"tasks": ".devteam/tasks.json"}


def test_langchain_tools_return_errors_as_text(ws):
    tools, missing = make_tools(ws, ["create_file", "run_tests"])
    assert [t.name for t in tools] == ["create_file"] and missing == ["run_tests"]
    reply = tools[0].invoke({"path": "../x.txt", "content": "x"})
    assert reply.startswith("ERROR:") and "outside the workspace" in reply
    bad_args = tools[0].invoke({"path": "x.txt"})
    assert bad_args.startswith("ERROR: invalid arguments")


def test_json_artifacts_validated_and_unwrapped(ws):
    import json

    escaped = json.dumps('[{"id": "T1"}]')  # JSON string containing JSON (one level too deep)
    ws.write_artifact("tasks.json", escaped)
    assert json.loads(ws.read_artifact("tasks.json")) == [{"id": "T1"}]
    with pytest.raises(ToolError, match="not valid JSON"):
        ws.write_artifact("tasks.json", "[{id: T1}]")
