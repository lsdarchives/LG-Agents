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

    # Outcome
    status: Literal["running", "success", "max_iterations"]
