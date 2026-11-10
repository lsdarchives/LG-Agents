# Tracing with MLflow (optional)

Step 10 adds MLflow tracing to the CLI. Normal runs do not need MLflow; install it only when you
want to inspect model calls and graph execution.

## Install MLflow

From the repository root, install the optional package into the project virtual environment:

```powershell
.venv\Scripts\python -m pip install mlflow
```

The main `requirements.txt` intentionally leaves MLflow optional. With the default local tracking
URI, MLflow stores run and trace data in `mlruns/`, which is already gitignored.

## Run with tracing

Pass `--trace` before the task description:

```powershell
.venv\Scripts\python main.py --trace "Write a function that merges overlapping intervals"
```

The CLI selects the `devteam` experiment, enables `mlflow.langchain.autolog()`, and starts one
MLflow run around the graph invocation. Autologging captures the LangGraph execution as a trace.
The CLI also logs these task-level values on the MLflow run:

| Value | MLflow field |
|---|---|
| Task, status, DevTeam run ID | Parameters |
| `iterations`, `best_score`, `test_fix_used`, `success` | Metrics |

Without `--trace`, the CLI works as before and never imports MLflow.

## Open the UI

Keep the CLI terminal available, open a second PowerShell window at the repository root, and run:

```powershell
.venv\Scripts\mlflow ui --backend-store-uri ./mlruns
```

Open `http://127.0.0.1:5000`. Select the `devteam` experiment. The **Runs** view shows the task
parameters and metrics. Open a run and use the **Traces** view to inspect the graph execution,
including nested agent calls and their inputs and outputs.

## Reading a trace

- The trace shows what ran and in what order; expand a node or LLM call to inspect its recorded
  inputs, output, and duration.
- Use the `status` parameter to distinguish a passing result from `max_iterations`.
- Compare `iterations` and `best_score` metrics across runs to see how many coder attempts were
  needed and how many tests the selected solution passed.
- The `test_fix_used` metric shows whether the one-time test rewrite route ran.
- MLflow runs and traces are separate views: the parameters and metrics appear on the run, while
  the detailed execution appears as a trace.

Autotracing records prompts, responses, state data, and generated solution code. Treat `mlruns/`
like the local `runs/` directory and keep it out of version control. To use a tracking server,
set `MLFLOW_TRACKING_URI` before invoking the CLI; use the same tracking URI when opening the UI.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `--trace` reports that MLflow is missing | The optional dependency is not installed in this venv | Run `.venv\Scripts\python -m pip install mlflow` |
| The experiment has no trace | The task did not reach an instrumented graph invocation, or tracing was not enabled | Run the command with `--trace` and check the CLI output for a failed run |
| Runs appear but traces are missing | Traces are not the same as MLflow run metrics | Open the experiment's **Traces** view; verify `mlflow.langchain.autolog()` ran |
| The UI shows no data | The UI is pointed at a different tracking store | Start it with `--backend-store-uri ./mlruns`, or configure the same `MLFLOW_TRACKING_URI` for both processes |

MLflow's LangChain integration can change independently of this project. If autologging reports a
compatibility warning, check the supported versions in the [MLflow LangChain autologging guide](https://mlflow.org/docs/latest/genai/flavors/langchain/autologging/).
