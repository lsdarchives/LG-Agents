# Tracing with MLflow (optional)

> **Stub: filled in at step 10 (optional).** Until then this page only describes what will go here.

Step 10 adds tracing, so every graph run records each LLM call (prompt, reply, timing) and each
node in a browsable UI.

## What this page will cover

- Installing MLflow and turning tracing on:
  `mlflow.set_experiment("devteam")` and `mlflow.langchain.autolog()`
- Running the dev team with tracing enabled
- Opening the UI with `mlflow ui` and finding a run
- Reading a trace: which node ran, what each agent was sent, why a round failed
- Keeping `mlruns/` out of git (it's already in `.gitignore`)
