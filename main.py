"""Command-line entry point for the DevTeam coding workflow."""

import argparse
import sys
from pathlib import Path

from devteam.state import initial_state


def _configure_mlflow():
    """Enable LangChain/LangGraph autotracing for this CLI invocation."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("Tracing requires MLflow; install it with `.venv\\Scripts\\python -m pip install mlflow`.") from exc

    mlflow.set_experiment("devteam")
    mlflow.langchain.autolog()
    return mlflow


def _log_mlflow_result(mlflow, task: str, result: dict) -> None:
    """Log task-level values alongside the LangGraph trace."""
    mlflow.log_param("task", task)
    mlflow.log_param("status", result["status"])
    mlflow.log_param("devteam_run_id", result["run_id"])
    mlflow.log_metrics(
        {
            "iterations": result["iteration"],
            "best_score": result["best_score"],
            "test_fix_used": int(result["test_fix_used"]),
            "success": int(result["status"] == "success"),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate, test, and iteratively fix a single-file Python solution."
    )
    parser.add_argument("task", help="Python task description, preferably in quotes")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="record an MLflow trace and task metrics (requires the optional mlflow package)",
    )
    args = parser.parse_args()

    try:
        mlflow = _configure_mlflow() if args.trace else None
        # Import after autolog setup so the LangGraph execution is instrumented.
        from devteam.graph import MAX_ITERATIONS, graph

        if mlflow is None:
            result = graph.invoke(initial_state(args.task))
        else:
            with mlflow.start_run(run_name="devteam"):
                result = graph.invoke(initial_state(args.task))
                _log_mlflow_result(mlflow, args.task, result)
    except KeyboardInterrupt:
        print("Run interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - report unexpected graph/provider failures cleanly
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    run_id = result["run_id"]
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    solution_path = run_dir / "solution.py"
    solution_path.write_text(result["code"], encoding="utf-8")

    print(f"\nStatus: {result['status']}")
    print(f"Coder attempts: {result['iteration']} / {MAX_ITERATIONS}")
    print(f"Best test score: {result['best_score']}")
    print(f"Test rewrite used: {result['test_fix_used']}")
    print(f"Run logs: {run_dir}")
    print(f"Final solution: {solution_path}")
    if result["test_output"]:
        print("\nPytest summary:")
        print(result["test_output"][-1200:])
    if result["lint_output"]:
        print("\nRuff output:")
        print(result["lint_output"])
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
