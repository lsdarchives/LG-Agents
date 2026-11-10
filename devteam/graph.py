"""LangGraph workflow for DevTeam, with a retry loop after failed checks."""

import uuid
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from devteam.agents import architect, coder, debugger, test_writer
from devteam.state import DevState
from devteam.tools import run_lint, run_tests

MAX_ITERATIONS = 5


def _write_run_log(state: DevState, run_id: str, test_output: str, lint_output: str) -> None:
    """Persist the current candidate, tests, diagnosis, and check output for this iteration."""
    iteration_dir = Path("runs") / run_id / f"iteration-{state['iteration']:02d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    (iteration_dir / "solution.py").write_text(state["code"], encoding="utf-8")
    (iteration_dir / "test_solution.py").write_text(state["tests"], encoding="utf-8")
    (iteration_dir / "diagnosis.txt").write_text(
        state.get("diagnosis") or "No diagnosis yet (first attempt).", encoding="utf-8"
    )
    (iteration_dir / "pytest.txt").write_text(test_output, encoding="utf-8")
    (iteration_dir / "ruff.txt").write_text(lint_output, encoding="utf-8")


def run_checks(state: DevState) -> dict:
    """Run checks, track the best candidate, set outcome and write iteration logs."""
    print("[run_checks] running pytest and ruff")
    test_result = run_tests(state["code"], state["tests"])
    lint_result = run_lint(state["code"])
    passed = test_result.passed and lint_result.clean
    best_score = state.get("best_score", 0)
    best_code = state.get("best_code", "")
    best_test_output = state.get("best_test_output", "")
    best_lint_output = state.get("best_lint_output", "")
    if passed or not best_code or test_result.score > best_score:
        best_code = state["code"]
        best_score = test_result.score
        best_test_output = test_result.output
        best_lint_output = lint_result.output

    run_id = state.get("run_id") or uuid.uuid4().hex[:12]
    _write_run_log(state, run_id, test_result.output, lint_result.output)
    at_cap = state["iteration"] >= MAX_ITERATIONS
    status = "success" if passed else "max_iterations" if at_cap else "running"
    final_code = best_code if status == "max_iterations" else state["code"]
    final_test_output = best_test_output if status == "max_iterations" else test_result.output
    final_lint_output = best_lint_output if status == "max_iterations" else lint_result.output
    print(
        f"[run_checks] tests_passed={test_result.passed} score={test_result.score} "
        f"lint_clean={lint_result.clean} passed={passed} status={status} "
        f"best_score={best_score} run_id={run_id}"
    )
    return {
        "test_output": final_test_output,
        "lint_output": final_lint_output,
        "passed": passed,
        "best_code": best_code,
        "best_score": best_score,
        "best_test_output": best_test_output,
        "best_lint_output": best_lint_output,
        "status": status,
        "code": final_code,
        "run_id": run_id,
    }


def route_after_checks(state: DevState) -> str:
    """Stop on success or at the attempt cap; otherwise ask the debugger."""
    return "done" if state["status"] != "running" else "debugger"


def route_after_debugger(state: DevState) -> str:
    """Rewrite faulty tests once; otherwise send the diagnosis to the coder."""
    if state.get("blame") == "test" and not state.get("test_fix_used", False):
        return "rewrite_tests"
    return "coder"


def rewrite_tests(state: DevState) -> dict:
    """Run the test writer and mark the one permitted test rewrite as used."""
    print("[rewrite_tests] applying the one-time test rewrite")
    return {**test_writer(state), "test_fix_used": True}


def build_graph():
    """Build the capped correction loop with a one-time test rewrite route."""
    builder = StateGraph(DevState)
    builder.add_node("architect", architect)
    builder.add_node("test_writer", test_writer)
    builder.add_node("coder", coder)
    builder.add_node("run_checks", run_checks)
    builder.add_node("debugger", debugger)
    builder.add_node("rewrite_tests", rewrite_tests)

    builder.add_edge(START, "architect")
    builder.add_edge("architect", "test_writer")
    builder.add_edge("test_writer", "coder")
    builder.add_edge("coder", "run_checks")
    builder.add_conditional_edges("run_checks", route_after_checks, {"done": END, "debugger": "debugger"})
    builder.add_conditional_edges(
        "debugger",
        route_after_debugger,
        {"rewrite_tests": "rewrite_tests", "coder": "coder"},
    )
    builder.add_edge("rewrite_tests", "coder")
    return builder.compile()


graph = build_graph()
