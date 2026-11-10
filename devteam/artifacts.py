"""Bridge between TeamState, the workspace's artifacts and an agent's `reads`.

Agents communicate only through artifacts (structured documents in workspace/.devteam/). Before an
activation, load_inputs() gathers exactly the agent's `reads`; afterwards sync_state() reports what
the workspace now holds so the graph state stays in step.
"""

import json
import shutil
from functools import cache

import yaml

from devteam.config import MAX_ACTIVATIONS, STACKS_FILE
from devteam.team_state import TeamState
from devteam.toolbox import ToolError, Workspace
from devteam.toolbox.workspace import ARTIFACT_ALIASES

# A `reads`/`writes` name whose artifact file is named differently.
ALIASES = ARTIFACT_ALIASES


def artifact_index(ws: Workspace) -> str:
    """One line per artifact: name, size and its first heading, so the Tech Lead can see progress."""
    lines = []
    for name, rel in ws.artifacts().items():
        text = (ws.root / rel).read_text(encoding="utf-8")
        first = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
        lines.append(f"- {name} ({rel.rsplit('/', 1)[-1]}, {len(text.splitlines())} lines): {first[:80]}")
    return "\n".join(lines) or "(no artifacts yet)"


@cache
def stack_catalog() -> str:
    """Stack profiles the architect may pick from: name, language, kind and use_when.

    Non-Python stacks run in Docker; without Docker only Python profiles are offered."""
    data = yaml.safe_load(STACKS_FILE.read_text(encoding="utf-8"))
    docker = shutil.which("docker") is not None
    stacks = [s for s in data["stacks"] if docker or s.get("language") == "python"]
    lines = [f"- {s['name']} ({s.get('language')}, {s.get('kind')}): {s.get('use_when', '')}" for s in stacks]
    note = "" if docker else "\n(Only Python stacks are available: Docker is not installed.)"
    return f"default: {data.get('default_stack')}\n" + "\n".join(lines) + note


@cache
def stack_profile(name: str | None = None) -> dict | None:
    """Return a configured stack profile, using the catalog default when no name is supplied."""
    data = yaml.safe_load(STACKS_FILE.read_text(encoding="utf-8"))
    selected = name or data.get("default_stack")
    return next((profile for profile in data["stacks"] if profile["name"] == selected), None)


def _read(ws: Workspace, name: str) -> str | None:
    try:
        return ws.read_artifact(ALIASES.get(name, name))
    except ToolError:
        return None


def load_inputs(reads: list[str], state: TeamState, ws: Workspace, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return {read_name: text} for an agent's `reads`. Unknown/missing reads are left out, and the
    builder shows them as "(not provided)". `extra` (e.g. the Tech Lead's handoff) wins over files."""
    current_task = next(
        (task for task in state.get("task_board", []) if task.get("id") == state.get("current_task")),
        None,
    )
    related_files = []
    test_files = []
    if current_task:
        for path in current_task.get("files", []):
            try:
                related_files.append(ws.view_file(path))
            except ToolError:
                continue
        test_path = current_task.get("test_file")
        if test_path:
            try:
                test_files.append(ws.view_file(test_path))
            except ToolError:
                pass
    profile = stack_profile(current_task.get("stack")) if current_task else None
    computed = {
        "request": state["request"],
        "task_board": json.dumps(state["task_board"], indent=1) if state["task_board"] else None,
        "task": json.dumps({**current_task, "id": current_task["id"]}, indent=2) if current_task else None,
        "stack_profile": json.dumps(profile, indent=2) if profile else None,
        "related_files": "\n\n".join(related_files) if related_files else None,
        "test_files": "\n\n".join(test_files) if test_files else None,
        "diagnosis": state.get("diagnosis") or None,
        "artifact_index": artifact_index(ws),
        "last_agent_result": json.dumps(state["last_result"]) if state["last_result"] else None,
        "budget": f"activations used: {state['activations']} of {MAX_ACTIVATIONS}",
        "stack_catalog": stack_catalog() if "stack_catalog" in reads else None,
    }
    inputs: dict[str, str] = {}
    for name in reads:
        value = (extra or {}).get(name) or computed.get(name) or _read(ws, name)
        if value:
            inputs[name] = value
    return inputs


def sync_state(ws: Workspace) -> dict:
    """State update reflecting the workspace: the artifact index, and the task board if tasks.json parses."""
    update: dict = {"artifacts": ws.artifacts()}
    raw = _read(ws, "task_board")
    if raw:
        try:
            tasks = json.loads(raw)
        except json.JSONDecodeError:
            tasks = None
        if isinstance(tasks, list) and all(isinstance(t, dict) and "id" in t for t in tasks):
            update["task_board"] = tasks
    return update
