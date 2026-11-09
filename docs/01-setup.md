# Setup

All commands run from the repo root in PowerShell.

## 1. Python version

Use **Python 3.14** (3.11+ works in principle).

Don't use 3.15 yet. `zstandard`, which langgraph's dependencies pull in, has no prebuilt wheel
for it, so pip tries to compile it and fails with
*"Microsoft Visual C++ 14.0 or greater is required"*.

```powershell
py -0p        # list installed Pythons
```

## 2. Virtual environment + dependencies

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Always call the venv's Python explicitly (`.venv\Scripts\python ...`), or activate it first
with `.venv\Scripts\Activate.ps1`.

## 3. API keys (`.env`)

Create `.env` in the repo root (it is gitignored, so never commit it):

```
GROQ_API_KEY=...          # required: primary provider (Groq), from console.groq.com
GOOGLE_API_KEY=...        # optional: Gemini fallback, from Google AI Studio
```

## 4. Checks per step

| Step | Check | Command / how |
|---|---|---|
| 1 | State imports cleanly | `.venv\Scripts\python -c "from devteam.state import DevState"` |
| 2 | Good code passes, bad fails, infinite loop times out | Run the **Try it** block in [LEARNING.md](LEARNING.md#step-2-the-test-and-lint-tools) |
| 3 | Architect writes a spec, test writer writes tests; tests pass against a known-good solution | Run the **Check** block in [LEARNING.md](LEARNING.md#step-3-architect--test-writer-standalone) with `.venv\Scripts\python -X utf8` |
| 4 | Coder's `solution.py` passes the test writer's tests and lints clean | Run the **Check** block in [LEARNING.md](LEARNING.md#step-4-coder-standalone) with `.venv\Scripts\python -X utf8` |
| 5 | Compile the linear graph | `.venv\Scripts\python -c "from devteam.graph import graph; print(sorted(graph.get_graph().nodes))"` (full LLM pass requires `.env`; invoke with a complete initial state) |
| 6 | Failed checks reach debugger and retry | Run a hard task such as the CSV parser from [03-tasks.md](03-tasks.md); inspect the node prints for `run_checks → debugger → coder → run_checks`. Requires `.env`. |
| 7 | Capped run retains best candidate and logs iterations | Temporarily set `MAX_ITERATIONS = 1`, run a task that fails its first check, then inspect final `status`, `best_score`, and `runs/<run_id>/iteration-01/` (solution, tests, diagnosis, pytest, ruff). Restore the cap to 5. Requires `.env`. |
| 8 | Test-blame route is limited to one rewrite | Run the route-guard command in [LEARNING.md](LEARNING.md#step-8-test-blame-route); a full test rewrite requires `.env`. |
| 9 | CLI runs a task and saves the selected solution | `.venv\Scripts\python main.py "Write a function that merges overlapping intervals"` (requires `.env`; see [LEARNING.md](LEARNING.md#step-9-cli)) |
| 10 | CLI records MLflow traces and task metrics | Install optional MLflow, run `.venv\Scripts\python main.py --trace "Write a function that merges overlapping intervals"`, then open the UI with `.venv\Scripts\mlflow ui`. See [05-tracing.md](05-tracing.md). |

### v2 checks

See [06-v2-guide.md](06-v2-guide.md) for each step's walkthrough.

| Step | Check | Command / how |
|---|---|---|
| 11 | agents.yaml validates; three agents reply with a valid result block | `.venv\Scripts\python -m devteam.registry` (no API calls), then `.venv\Scripts\python -X utf8 -m devteam.builder` (requires `.env`; expect `step 11 check: PASS`) |

Lint the project code itself:

```powershell
.venv\Scripts\python -m ruff check devteam
```
