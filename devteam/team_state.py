"""v2 shared state. Reducers let several nodes contribute to the same field safely.

(Named team_state.py because v1 owns devteam/state.py.)
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages


class Task(TypedDict, total=False):
    id: str
    title: str
    owner_role: str
    component: str
    stack: str
    files: list[str]
    depends_on: list[str]
    acceptance: list[str]
    test_file: str
    status: Literal["todo", "in_progress", "done", "failed"]
    fix_loops: int
    _delete: bool  # internal reducer marker used when a human removes a task from the plan


def merge_dicts(old: dict[str, str] | None, new: dict[str, str] | None) -> dict[str, str]:
    """Reducer: later keys win; nothing is ever dropped."""
    return {**(old or {}), **(new or {})}


def merge_tasks(old: list[Task] | None, new: list[Task] | None) -> list[Task]:
    """Reducer: merge tasks by id. An update for an existing id changes only the fields it sends;
    new ids are appended in order. Two parallel builders can each update their own task."""
    merged: dict[str, Task] = {t["id"]: dict(t) for t in (old or [])}
    for task in new or []:
        if task.get("_delete"):
            merged.pop(task["id"], None)
            continue
        merged[task["id"]] = {**merged.get(task["id"], {}), **task}
    return list(merged.values())


class TeamState(TypedDict):
    request: str
    run_id: str
    messages: Annotated[list, add_messages]  # audit log of agent results (not shared context)
    artifacts: Annotated[dict[str, str], merge_dicts]  # artifact name -> path in the workspace
    task_board: Annotated[list[Task], merge_tasks]
    current_task: str | None
    diagnosis: str
    handoff: dict | None  # objective, inputs, output and boundaries for the current assignment
    last_result: dict  # last agent's result block + its name
    activations: Annotated[int, operator.add]  # each node returns 1; the reducer sums
    review_rounds: int
    status: Literal["running", "awaiting_approval", "done", "partial", "failed"]
    plan_approved: bool
    reason: str


def initial_team_state(request: str, run_id: str) -> TeamState:
    return {
        "request": request,
        "run_id": run_id,
        "messages": [],
        "artifacts": {},
        "task_board": [],
        "current_task": None,
        "diagnosis": "",
        "handoff": None,
        "last_result": {},
        "activations": 0,
        "review_rounds": 0,
        "status": "running",
        "plan_approved": False,
        "reason": "",
    }
