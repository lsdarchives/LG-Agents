"""Shared state passed between every node in the DevTeam graph."""

from typing import Literal, TypedDict


class DevState(TypedDict):
    # Input
    task: str  # the user's task description

    # Artifacts produced by agents
    spec: str  # architect's spec: signatures, behaviour, edge cases, examples
    tests: str  # contents of test_solution.py (test_writer)
    code: str  # contents of solution.py (coder)

    # Results of the last run_checks
    test_output: str  # pytest output (or "TIMEOUT")
    lint_output: str  # ruff output
    passed: bool  # True only if tests pass AND lint is clean

    # Debugger verdict
    diagnosis: str  # what went wrong and how to fix it
    blame: Literal["code", "test"] | None  # who is at fault; None before first debug

    # Loop control
    iteration: int  # number of coder attempts so far (coder increments)
    test_fix_used: bool  # the one-time "rewrite tests" route has been taken

    # Best attempt tracking
    best_code: str  # code with the highest score so far
    best_score: int  # number of passed tests for best_code
    best_test_output: str  # pytest output corresponding to best_code
    best_lint_output: str  # ruff output corresponding to best_code
    run_id: str  # directory name used for this run's local logs

    # Outcome
    status: Literal["running", "success", "max_iterations"]


def initial_state(task: str) -> DevState:
    """Create an empty graph state for one CLI task."""
    return {
        "task": task,
        "spec": "",
        "tests": "",
        "code": "",
        "test_output": "",
        "lint_output": "",
        "passed": False,
        "diagnosis": "",
        "blame": None,
        "iteration": 0,
        "test_fix_used": False,
        "best_code": "",
        "best_score": 0,
        "best_test_output": "",
        "best_lint_output": "",
        "run_id": "",
        "status": "running",
    }
