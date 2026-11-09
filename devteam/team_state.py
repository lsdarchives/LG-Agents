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
    status: Literal["todo", "in_progress", "done", "failed"]


def merge_dicts(old: dict[str, str] | None, new: dict[str, str] | None) -> dict[str, str]:
    """Reducer: later keys win; nothing is ever dropped."""
    return {**(old or {}), **(new or {})}


def merge_tasks(old: list[Task] | None, new: list[Task] | None) -> list[Task]:
    """Reducer: merge tasks by id. An update for an existing id changes only the fields it sends;
    new ids are appended in order. Two parallel builders can each update their own task."""
    merged: dict[str, Task] = {t["id"]: dict(t) for t in (old or [])}
    for task in new or []:
        merged[task["id"]] = {**merged.get(task["id"], {}), **task}
    return list(merged.values())


class TeamState(TypedDict):
    request: str
    run_id: str
    messages: Annotated[list, add_messages]  # audit log of agent results (not shared context)
    artifacts: Annotated[dict[str, str], merge_dicts]  # artifact name -> path in the workspace
    task_board: Annotated[list[Task], merge_tasks]
    current_task: str | None
    last_result: dict  # last agent's result block + its name
    activations: Annotated[int, operator.add]  # each node returns 1; the reducer sums
    review_rounds: int
    status: Literal["running", "done", "partial", "failed"]
    reason: str


def initial_team_state(request: str, run_id: str) -> TeamState:
    return {
        "request": request,
        "run_id": run_id,
        "messages": [],
        "artifacts": {},
        "task_board": [],
        "current_task": None,
        "last_result": {},
        "activations": 0,
        "review_rounds": 0,
        "status": "running",
        "reason": "",
    }
