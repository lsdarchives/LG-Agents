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
| 5+ | See the build order in [../CLAUDE.md](../CLAUDE.md) | Commands are added here as each step lands |

Lint the project code itself:

```powershell
.venv\Scripts\python -m ruff check devteam
```
