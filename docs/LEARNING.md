# Build a Self-Correcting AI Coding Team with LangGraph

A hands-on guide to building a multi-agent system that takes a programming task, writes tests,
writes code, **runs the tests for real**, and keeps fixing its own mistakes until the tests pass
or it hits a limit.

You'll learn LangGraph by building something real, and come away with a pattern
(generate → verify → diagnose → retry) that you can reuse well beyond coding agents.

---

## Contents

- [Who this guide is for](#who-this-guide-is-for)
- [What you'll build](#what-youll-build)
- [Prerequisites and setup](#prerequisites-and-setup)
- [Part 1: Core concepts](#part-1-core-concepts)
- [Part 2: System architecture](#part-2-system-architecture)
- [Part 3: Build it, step by step](#part-3-build-it-step-by-step)
- [Part 4: Adapting the pattern to your own systems](#part-4-adapting-the-pattern-to-your-own-systems)
- [Part 5: FAQ](#part-5-faq)
- [Troubleshooting](#troubleshooting)
- [Glossary](#glossary)

---

## Who this guide is for

- You're comfortable with Python: functions, dicts, type hints, virtual environments.
- You've used an LLM API or chat model at least once, or are happy to learn as you go.
- You want to understand **how agent systems work under the hood**, not just call a framework
  that hides everything.

No prior LangGraph or LangChain experience is needed.

## What you'll build

A command-line tool:

```
python main.py "Write a function that merges overlapping intervals"
```

Behind the scenes, a team of four LLM "agents" and one plain-code checker work together:

1. An **architect** turns the task into a precise spec.
2. A **test writer** turns the spec into a pytest file.
3. A **coder** writes `solution.py` to pass those tests.
4. A **checker** (no LLM) actually runs pytest and ruff.
5. If anything fails, a **debugger** works out what went wrong and who's to blame (the code or the
   test), and sends the work back to the right agent.

It stops when the tests pass and the code is lint-clean, or after 5 attempts. In that case it hands
back the best attempt it produced.

**Scope (v1):** single-file Python tasks, one LLM model for all agents, CLI only.

## Prerequisites and setup

- Python 3.11–3.14. **Avoid 3.15 for now:** `zstandard`, which langgraph's dependencies pull in,
  has no prebuilt wheel for it yet, so pip tries to compile it and fails (on Windows with
  *"Microsoft Visual C++ 14.0 or greater is required"*).
- A free Google AI Studio API key (Gemini). Groq is an optional fallback.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install langgraph langchain-google-genai pytest ruff python-dotenv
```

Create `.env` in the project root and **never commit it**:

```
GOOGLE_API_KEY=your-key-here
```

The line must be exactly `NAME=value`: no spaces around `=`, no quotes. A key pasted on its own
line is silently ignored, and you'll get *"No API key was provided"* later.

`.gitignore`:

```
.env
.venv/
runs/
mlruns/
```

Project layout you'll end up with:

```
main.py            # CLI entry point
runs/              # per-run logs (gitignored)
docs/
  03-tasks.md      # sample tasks to test the system with
devteam/
  __init__.py
  state.py         # the shared state definition
  prompts.py       # all LLM prompts
  agents.py        # architect, test_writer, coder, debugger
  tools.py         # run_tests, run_lint (no LLM)
  graph.py         # wires everything into a LangGraph graph
```

---

## Part 1: Core concepts

Read this part once before you start building, then come back to it whenever a step mentions a
concept. Each entry gives a **definition**, **why it matters**, and **where it shows up** in the
project.

### 1.1 Agent

**Definition:** a component that uses an LLM to decide what to produce or do next, based on
instructions (a prompt) and inputs.

**Why it matters:** "agent" is used loosely in the industry. In this guide, an agent is simply
*a function that calls an LLM with a focused role*. There's no magic autonomy.

**In the project:** architect, test_writer, coder, debugger.

### 1.2 Multi-agent system

**Definition:** several agents, each with a narrow role, cooperating on one task by passing work
between them.

**Why it matters:** one huge prompt that tries to "spec, test, code and debug" does all four
badly. Narrow roles mean shorter prompts, clearer outputs, and failures you can pin on one part.

**In the project:** four LLM roles plus one tool node, connected by a graph.

### 1.3 LLM node vs tool node

**Definition:** an **LLM node** calls a model. A **tool node** runs ordinary deterministic code,
like running a test suite, querying a database or calling an API.

**Why it matters:** this is the most important design idea in the guide. **LLMs generate, tools
verify.** An LLM can claim its code works. pytest *proves* it. Systems that check LLM output with
real tools are far more reliable than systems that ask the LLM to check itself.

**In the project:** `run_checks` is a tool node, and every other node is an LLM node.

### 1.4 Ground truth

**Definition:** a signal that tells you objectively whether output is correct, independent of
the model's opinion.

**Why it matters:** a retry loop is only as good as its feedback. With ground truth (test
results), each retry gets real error messages to fix. Without it, the model is just guessing again.

**In the project:** pytest output and the number of passed tests, plus ruff lint output.

### 1.5 Graph, node, edge

**Definition:** LangGraph models your program as a **graph**:
- A **node** is a Python function: it takes the state and returns an update.
- An **edge** is a fixed connection: "after node A, run node B".
- `START` and `END` are special markers for where execution begins and stops.

**Why it matters:** the graph *is* your control flow. You can read it, draw it and change it
without rewriting the agents.

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(MyState)
builder.add_node("architect", architect_node)
builder.add_node("coder", coder_node)
builder.add_edge(START, "architect")
builder.add_edge("architect", "coder")
builder.add_edge("coder", END)
graph = builder.compile()
```

**In the project:** step 5 builds the first graph.

### 1.6 State

**Definition:** a single shared data structure (usually a `TypedDict`) that holds everything the
graph knows. Every node reads from it and writes to it.

**Why it matters:** nodes never call each other directly or pass arguments to each other. State
is the **only** way information moves through the graph. If a field isn't in the state, the next
node can't see it.

**In the project:** `DevState` in `devteam/state.py` (step 1).

### 1.7 Partial update and reducer

**Definition:** a node returns a dict containing **only the fields it changed**, for example
`{"code": new_code}`. LangGraph merges that into the state using a **reducer** for each field.
The default reducer is *replace* (last write wins).

**Why it matters:** nodes stay small, and you never copy the whole state around. If you want a
field to *accumulate* instead of being replaced (a message history, say), you attach a reducer:

```python
import operator
from typing import Annotated, TypedDict

class ChatState(TypedDict):
    messages: Annotated[list, operator.add]  # each update is appended, not replaced
```

**In the project:** every field uses the default *replace* reducer.

### 1.8 Conditional edge (routing)

**Definition:** an edge whose destination is chosen **at runtime** by a routing function that
reads the state and returns the name of the next node.

```python
def route_after_checks(state) -> str:
    if state["passed"]:
        return "done"
    return "debugger"

builder.add_conditional_edges(
    "run_checks",
    route_after_checks,
    {"done": END, "debugger": "debugger"},
)
```

**Why it matters:** this is how a graph makes decisions. Keep routing functions **pure and
cheap**: they only read the state and never call the LLM.

**In the project:** after `run_checks` (pass, give up, or debug) and after `debugger` (fix code
or fix tests). Built in step 6.

### 1.9 Cycle (loop)

**Definition:** a path through the graph that leads back to an earlier node. In LangGraph, a loop
is just a conditional edge that points backwards.

**Why it matters:** loops are what let a system *self-correct*. LangGraph supports them directly,
unlike the older LangChain "chains", which only ran in one direction. Nothing loops unless you
build the edge yourself.

**In the project:** `run_checks → debugger → coder → run_checks`.

### 1.10 Termination guarantee

**Definition:** a rule that makes sure every loop eventually stops.

**Why it matters:** an LLM might *never* fix a bug. Without a cap, a loop runs until it burns your
API quota. Always have:
1. **Your own cap** in the state (e.g. `iteration >= 5` → stop), with a meaningful outcome.
2. **LangGraph's recursion limit** as a safety net. The default is 25 steps, and hitting it raises
   `GraphRecursionError`. You can change it with `graph.invoke(state, {"recursion_limit": 50})`.

**In the project:** `iteration` cap plus `status="max_iterations"` (step 7).

### 1.11 Best-attempt tracking

**Definition:** keeping the best result seen so far, scored objectively, so that a run which never
fully succeeds still returns something useful.

**Why it matters:** retries don't always improve things. Attempt 3 can be worse than attempt 2.
Score every attempt and keep the best.

**In the project:** `best_code` / `best_score`, where score = number of passing tests (step 7).

### 1.12 Structured output

**Definition:** making the LLM return data that fits a schema (a Pydantic model) instead of free
text.

```python
from typing import Literal
from pydantic import BaseModel

class Diagnosis(BaseModel):
    diagnosis: str
    blame: Literal["code", "test"]
    fix_hint: str

debugger_llm = llm.with_structured_output(Diagnosis)
result = debugger_llm.invoke(prompt)  # -> Diagnosis instance
```

**Why it matters:** routing needs a reliable value to branch on. `if "test" in response_text`
is fragile, while `result.blame == "test"` is not.

**In the project:** the debugger (step 6).

### 1.13 Context engineering (lean prompts)

**Definition:** deciding exactly what information each LLM call receives.

**Why it matters:** more context is **not** better. Sending the whole history costs tokens, hits
rate limits sooner and distracts the model. Send each agent only what its role needs: the coder on
a retry gets the spec, the current code, the last test output and the diagnosis, and nothing else.

**In the project:** every prompt in `prompts.py`. State holds everything, and each node picks what
to send.

### 1.14 Rate limits and backoff

**Definition:** free-tier APIs limit requests per minute and per day. When you exceed them they
return HTTP 429 / `RESOURCE_EXHAUSTED`. A busy model can also return 503 `UNAVAILABLE`.
**Exponential backoff** means waiting progressively longer between retries (10s, 20s, 40s) before
giving up with a clear error.

**Why it matters:** a multi-agent loop makes many calls quickly. Without backoff, one rate-limit
error crashes the whole run.

Retry only **transient** errors (429, 503). A wrong key (401) or an unknown model (404) will fail
every time, so retrying them just wastes a minute. And keep **one** retry layer: LangChain's
model classes retry on their own by default, so turn that off (`max_retries=0`) when you write
your own, or the waits multiply.

Not every 429 is the same, either. A **per-minute** limit clears in seconds, so backoff works. A
**per-day** limit (on Gemini's free tier this can be as low as 20 requests per model per day)
won't clear for hours, so fail at once and say so. The error text names the quota
(`...PerDay...`), which tells you which one you hit.

**In the project:** `call_llm()` in `devteam/agents.py` wraps every LLM call (step 3).

### 1.15 Provider abstraction

**Definition:** creating the model in one place (`get_llm()`) so every agent uses it and switching
provider or model is a one-line change.

**Why it matters:** models get renamed and deprecated, and free tiers change. On a given day the
newest model can also be so overloaded that a one-word reply takes 90 seconds, while an older one
answers in 3. You'll switch more often than you expect.

**In the project:** `MODEL_NAME` and `get_llm()` in `devteam/agents.py`. Gemini is the default,
and Groq (`langchain-groq`) is the fallback.

### 1.16 Role boundaries

**Definition:** hard rules about what each agent may change.

**Why it matters:** if the coder could edit the tests, the easiest way to "pass" would be to weaken
them. Generators must never be able to change their own grading criteria.

**In the project:** the coder only ever outputs `solution.py`. Only the test writer writes tests,
and it gets **at most one** retry, and only when the debugger blames the test.

### 1.17 Sandboxing (and its limits)

**Definition:** running untrusted code somewhere it can't cause harm.

**Why it matters:** LLM-generated code is untrusted. This guide uses a **temp directory and a
subprocess with a timeout**. That stops leftover files and infinite loops, but it is **not a
security sandbox**. The code still runs with your user's permissions. For anything beyond your own
practice tasks, use containers (Docker) or a dedicated sandbox service.

**In the project:** `devteam/tools.py` (step 2).

---

## Part 2: System architecture

### The flow

```
            ┌──────────────────────────────────────────────┐
            │                                              │
task ──► architect ──► test_writer ──► coder ──► run_checks ──► END (success / max_iterations)
                          ▲              ▲          │
                          │              │          ▼
                          └── blame=test ┴─ debugger
                             (once only)    blame=code
```

### Roles

| Role | Type | Input | Output |
|---|---|---|---|
| architect | LLM | task | spec: function signatures, behaviour, edge cases, examples. **No code.** |
| test_writer | LLM | spec | one pytest file that imports from `solution` |
| coder | LLM | spec (+ diagnosis + test output on retries) | `solution.py` only, **never tests** |
| run_checks | tool | code + tests | pytest + ruff results, score |
| debugger | LLM, structured | spec, code, tests, test output | `diagnosis`, `blame` (`code`/`test`), `fix_hint` |

### Routing rules

After **run_checks**:
- tests pass **and** lint is clean → `END`, `status="success"`
- `iteration >= 5` → `END`, `status="max_iterations"`, return `best_code`
- otherwise → `debugger`

After **debugger**:
- `blame == "test"` and the test fix hasn't been used yet → `test_writer` (set `test_fix_used=True`)
- otherwise → `coder`

The **coder** increments `iteration` each time it runs.

### Design principles to carry into your own systems

1. **Verify with tools, not with LLMs.**
2. **Narrow roles, lean prompts.**
3. **Generators can't edit their graders.**
4. **Every loop has a cap and a fallback result.**
5. **Route on structured fields, not on free text.**
6. **Build and test each piece alone before wiring them together.**

---

## Part 3: Build it, step by step

Each step has the same parts: **Goal**, **Concepts**, **Code**, **Design notes**, **Check**,
**Try it**, **Check yourself**, **Gotchas**. Don't skip the checks. Each step assumes the
previous one works.

Build order:

| Step | What | Check |
|---|---|---|
| 1 | Setup + state | imports cleanly |
| 2 | Test and lint tools | good code passes, bad code fails, infinite loop times out |
| 3 | Architect + test writer, standalone | read their outputs |
| 4 | Coder, standalone | its code passes the tests when run manually |
| 5 | Linear graph (no loop) | one full pass end-to-end |
| 6 | Debugger + conditional edges | a hard task iterates |
| 7 | Cap + best attempt + run logs | cap=1 forces a stop |
| 8 | Test-blame route | a planted wrong test is rerouted once |
| 9 | CLI | all sample tasks run; record iterations and pass rate |
| 10 | (Optional) MLflow tracing | traces visible in `mlflow ui` |

Why does the loop only arrive in step 6? If you wire everything up on day one and get bad output,
you can't tell whether a prompt, the routing or the state is at fault. By step 6, the loop is the
only new thing.

---

### Step 1: Setup + state

**Goal:** define the single data structure every node will share.

**Concepts:** [State](#16-state), [Partial update and reducer](#17-partial-update-and-reducer).

**Code:** `devteam/state.py`

```python
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
```

Also create an empty `devteam/__init__.py`.

**Design notes**
- **Why `TypedDict`?** It's LangGraph's standard state type and it's a plain dict at runtime, so
  nodes can just `return {"code": "..."}`. The type hints help you and your editor, but **Python
  doesn't check them at runtime.**
- **Design state by asking "who writes what":**

  | Group | Fields | Written by |
  |---|---|---|
  | Input | `task` | CLI |
  | Artifacts | `spec`, `tests`, `code` | architect, test_writer, coder |
  | Check results | `test_output`, `lint_output`, `passed` | run_checks |
  | Debugger verdict | `diagnosis`, `blame` | debugger |
  | Loop control | `iteration`, `test_fix_used` | coder, debugger route |
  | Best attempt | `best_code`, `best_score` | run_checks |
  | Outcome | `status` | the end of the graph |

- **Loop-control fields exist *because* graphs can loop.** Plan for termination from the
  first file you write.

**Check:** `python -c "from devteam.state import DevState; print('ok')"`

**Try it**
```python
from typing import get_type_hints
from devteam.state import DevState

print(get_type_hints(DevState))   # every field and its type

s: DevState = {"task": 123}        # wrong type, missing fields...
print(s)                           # ...runs anyway: TypedDict is not validated
```

**Check yourself**
1. A node returns `{"code": "x = 1"}`. What happens to `tests` in the state?
2. Why is `blame` allowed to be `None`?
3. Which field stops the graph from looping forever?

<details><summary>Answers</summary>

1. Nothing. It keeps its previous value. Only the fields a node returns are updated.
2. Before the debugger has run for the first time there is no verdict yet.
3. `iteration`, checked against the cap. `test_fix_used` limits the test-rewrite route to once.
</details>

**Gotchas**
- A typo in a returned key (`{"cod": ...}`) is an easy bug to miss. Print node outputs while you develop.

---

### Step 2: The test and lint tools

**Goal:** run generated code and tests for real, safely enough for practice, and return an
objective result.

**Concepts:** [LLM node vs tool node](#13-llm-node-vs-tool-node),
[Ground truth](#14-ground-truth), [Sandboxing](#117-sandboxing-and-its-limits).

**Code:** `devteam/tools.py`

```python
"""Tool functions (no LLM): run pytest and ruff on generated code in a temp dir.

Not real isolation - generated code runs with your user's permissions.
Only run tasks you wrote yourself.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

TIMEOUT_SECONDS = 30

SUBPROCESS_ENV = {
    **os.environ,
    # Don't auto-load third-party pytest plugins (e.g. langsmith's), whose warnings
    # would clutter the output the debugger reads. Generated tests only need pytest.
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONUTF8": "1",  # non-ASCII test strings print correctly on Windows
}


class TestResult(NamedTuple):
    output: str  # pytest output, or "TIMEOUT ..." if it ran too long
    passed: bool  # True only if every test passed (exit code 0)
    score: int  # number of passed tests, parsed from the summary line


class LintResult(NamedTuple):
    output: str  # ruff output
    clean: bool  # True if ruff found no problems


def _run(cmd: list[str], cwd: Path) -> tuple[int | None, str]:
    """Run a command; return (exit code, combined output). Exit code None = timeout."""
    try:
        proc = subprocess.run(
            cmd,
            check=False,  # non-zero exit = failing tests/lint, which is normal here
            cwd=cwd,
            env=SUBPROCESS_ENV,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT: command took longer than {TIMEOUT_SECONDS}s"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def count_passed(output: str) -> int:
    """Parse e.g. '3 failed, 5 passed in 0.12s' -> 5."""
    match = re.search(r"(\d+) passed", output)
    return int(match.group(1)) if match else 0


def run_tests(code: str, tests: str) -> TestResult:
    """Write solution.py + test_solution.py to a fresh temp dir and run pytest."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        (workdir / "test_solution.py").write_text(tests, encoding="utf-8")
        cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"]
        returncode, output = _run(cmd, workdir)

    if returncode is None:
        return TestResult(output=output, passed=False, score=0)
    # Exit code 0 = all passed. (5 = no tests collected, which also counts as failure.)
    return TestResult(output=output, passed=returncode == 0, score=count_passed(output))


def run_lint(code: str) -> LintResult:
    """Run `ruff check solution.py` in a fresh temp dir, ignoring any project config."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        cmd = [sys.executable, "-m", "ruff", "check", "--isolated", "--no-cache", "solution.py"]
        returncode, output = _run(cmd, workdir)

    return LintResult(output=output, clean=returncode == 0)
```

**Design notes**
- **These are plain functions, not graph nodes yet.** In step 5 a small `run_checks` node will
  call them and write the results into the state. Keeping the logic separate means you can test it
  without LangGraph.
- **A fresh temp dir per run:** no leftover files from earlier attempts can affect a result.
- **`sys.executable -m pytest`:** uses the *current venv's* pytest, not whatever is first on PATH.
- **The timeout counts as a failure:** LLMs do write `while True:`. The subprocess is killed after
  30s and the result says `TIMEOUT`.
- **Exit codes:** pytest returns `0` only when all tests pass. `5` means *no tests collected*,
  for example an empty or broken test file. Treat it as a failure, never as "nothing failed".
- **Score = number of passing tests:** an all-or-nothing exit code can't show progress. The score
  can, and best-attempt tracking needs it.
- **`ruff check --isolated`:** ignores any ruff config in parent folders, so results are
  reproducible.
- **Results as `NamedTuple`:** named fields (`result.passed`) are clearer than a bare tuple.

**Check:** good code passes, bad code fails, an infinite loop times out (see below).

**Try it**
```python
from devteam.tools import run_tests, run_lint

tests = "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"

print(run_tests("def add(a, b):\n    return a + b\n", tests))   # passed=True, score=1
print(run_tests("def add(a, b):\n    return a - b\n", tests))   # passed=False, score=0
print(run_tests("def add(a, b):\n    return a + b\n", ""))      # exit code 5: passed=False
print(run_lint("import os\n"))                                   # F401 unused import, clean=False

# Takes ~30s, so you can watch the timeout kick in:
print(run_tests("def add(a, b):\n    while True:\n        pass\n", tests).output)
```

**Check yourself**
1. Why use a new temp dir every run instead of one fixed `work/` folder?
2. The LLM writes a test file with a syntax error. What do `passed` and `score` come back as?
3. Why parse a score when the exit code already says pass or fail?

<details><summary>Answers</summary>

1. Isolation between attempts: no stale `solution.py`, `__pycache__` or cache files can change the
   outcome.
2. `passed=False` (collection error gives a non-zero exit code), `score=0` (no "N passed" in the output).
3. The exit code is all-or-nothing. The score shows partial progress, which we need to pick
   the best attempt when the cap is hit.
</details>

**Gotchas**
- Windows sometimes can't delete a temp file that another process still has open. That's why we
  pass `ignore_cleanup_errors=True`.
- `-p no:cacheprovider` and `--no-cache` stop pytest and ruff from leaving cache folders behind.
- **Installed packages can inject pytest plugins.** `langsmith` (a LangChain dependency)
  registers one that prints a dozen deprecation warnings on every run. That noise ends up in
  `test_output`, which the debugger will read, so `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` turns
  third-party plugins off for the generated tests.

---

### Step 3: Architect + test writer, standalone

**Goal:** two LLM functions you can call by hand: task → spec, and spec → pytest file.

**Concepts:** [Agent](#11-agent), [Context engineering](#113-context-engineering-lean-prompts),
[Rate limits and backoff](#114-rate-limits-and-backoff),
[Provider abstraction](#115-provider-abstraction).

**Code:** `devteam/prompts.py` holds every prompt, so you can tune wording without touching logic.

```python
"""All LLM prompts live here, not inline in agents.py."""

ARCHITECT_SYSTEM = """\
You are a software architect. Turn a small Python task into a precise spec that a
test writer and a coder can work from independently.

The solution will be a single module named `solution.py`, standard library only.

Write the spec in Markdown with exactly these sections:
## Functions
Each public function's exact signature with type hints, and a one-line purpose.
## Behaviour
Precise rules, including how input is normalised and what is returned.
## Edge cases
Empty input, boundaries, unusual but valid input, and what each returns.
## Errors
Which invalid inputs raise which exception type (e.g. ValueError). Be specific.
## Examples
At least 6 concrete `call -> result` lines, covering normal and edge cases.

Rules:
- Do NOT write any implementation code.
- Be unambiguous: the tests and the code will be written from this spec alone.
- Keep it short; do not add features the task did not ask for.
"""

ARCHITECT_USER = "Task: {task}"

TEST_WRITER_SYSTEM = """\
You are a test engineer. Write a pytest test file for the spec you are given.

Rules:
- Output ONLY the Python code of `test_solution.py`, no explanations.
- Import the functions under test with `from solution import ...`.
- Test only behaviour stated in the spec: every example, every edge case, every error.
- Use plain `assert`, `pytest.raises` for errors, and `pytest.mark.parametrize` for tables.
- Each expected value must follow directly from the spec. If the spec is unclear on a
  case, leave that case out rather than guess.
- Standard library + pytest only; no files, network, randomness or time-dependence.
- Aim for 8-20 focused tests.
"""

TEST_WRITER_USER = "Spec:\n\n{spec}"
```

`devteam/agents.py` holds the model setup, the retry wrapper and the first two agents.

````python
"""LLM agent nodes. Each node: state -> partial update dict."""

import re
import time
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from devteam import prompts
from devteam.state import DevState

# Change the model here only. If a model is slow or overloaded, try another Flash model.
MODEL_NAME = "gemini-3.5-flash"
RETRY_WAITS = [10, 20, 40]  # seconds to wait after each retryable error


@lru_cache(maxsize=1)
def get_llm():
    """Create the chat model once. Switching provider is a change here only."""
    load_dotenv()  # reads GOOGLE_API_KEY from .env
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.2,
        thinking_level="low",  # Gemini 3's default (high) is many times slower; low is enough here
        timeout=120,  # seconds per request, so a hung call fails instead of waiting forever
        max_retries=0,  # we do our own backoff in call_llm, so retries don't stack up
    )


def _should_retry(exc: Exception) -> bool:
    """True for temporary 'slow down' errors: 429 quota and 503 model overloaded."""
    text = str(exc)
    return isinstance(exc, ModelRateLimitError) or any(
        code in text for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")
    )


def _is_daily_quota(exc: Exception) -> bool:
    """True if the 429 is the per-day free-tier cap: waiting seconds won't help."""
    return "PerDay" in str(exc) or "per day" in str(exc).lower()


def call_llm(system: str, user: str) -> str:
    """Send one system + user message, retrying on 429/503. Returns the reply text."""
    messages = [SystemMessage(system), HumanMessage(user)]
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            return get_llm().invoke(messages).text
        except Exception as exc:
            if not _should_retry(exc):
                raise
            if _is_daily_quota(exc):
                raise RuntimeError(
                    f"Daily free-tier quota for {MODEL_NAME} is used up; retrying today won't "
                    "help. Switch MODEL_NAME to another model, use the fallback provider, "
                    "or wait for the reset."
                ) from exc
            if attempt == len(RETRY_WAITS):
                raise RuntimeError(
                    f"Still rate-limited/overloaded after {len(RETRY_WAITS)} retries "
                    f"(free-tier quota used up, or model busy?). Last error: {exc}"
                ) from exc
            wait = RETRY_WAITS[attempt]
            print(f"  [retry] {type(exc).__name__}: waiting {wait}s ({attempt + 1}/{len(RETRY_WAITS)})")
            time.sleep(wait)
    raise AssertionError("unreachable")


def strip_code_fences(text: str) -> str:
    """Return the code inside ```python ... ``` fences, or the text itself if unfenced."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip() + "\n"


def _preview(text: str, lines: int = 6) -> str:
    head = text.strip().splitlines()[:lines]
    return "\n".join(f"    | {line}" for line in head)


def architect(state: DevState) -> dict:
    """task -> spec (signatures, behaviour, edge cases, errors, examples). No code."""
    print(f"[architect] task={state['task']!r}")
    spec = call_llm(prompts.ARCHITECT_SYSTEM, prompts.ARCHITECT_USER.format(task=state["task"]))
    print(f"[architect] spec: {len(spec.splitlines())} lines\n{_preview(spec)}")
    return {"spec": spec}


def test_writer(state: DevState) -> dict:
    """spec -> test_solution.py source."""
    print("[test_writer] writing tests from spec")
    reply = call_llm(prompts.TEST_WRITER_SYSTEM, prompts.TEST_WRITER_USER.format(spec=state["spec"]))
    tests = strip_code_fences(reply)
    n_tests = len(re.findall(r"^\s*def test_", tests, re.MULTILINE))
    print(f"[test_writer] {n_tests} test functions, {len(tests.splitlines())} lines")
    return {"tests": tests}
````

**Design notes**
- **Agents are already graph nodes.** `architect` and `test_writer` have the exact node shape:
  take `DevState`, return a *partial update* (`{"spec": ...}`). They don't import LangGraph and
  don't know what runs next. In step 5 you register them with `add_node` unchanged.
- **One `get_llm()`, created once.** `@lru_cache` makes it a lazily created singleton: the first
  call builds the client, later calls reuse it. Every agent goes through it, so switching model or
  provider is a one-line change.
- **System vs user message.** The *system* message is the role and the rules (fixed per agent).
  The *user* message is this run's input (the task, the spec). Keeping them apart makes prompts
  reusable and keeps each one short.
- **The spec is the contract.** The test writer and (in step 4) the coder both work from the spec
  *alone*, never from each other's output. If the spec is vague, they'll disagree, so the architect
  prompt forces fixed sections: Functions, Behaviour, Edge cases, Errors, Examples.
- **"If unsure, leave it out."** A wrong test is worse than a missing one: it makes correct code
  fail and sends the loop chasing a bug that doesn't exist. The test writer is told to skip cases
  the spec doesn't settle.
- **Backoff you can see.** `call_llm` retries only *temporary* errors (429 quota, 503 overloaded)
  and waits 10s, 20s, then 40s before giving up with a clear message. The library's own retries
  are switched off (`max_retries=0`); otherwise both layers retry and the waits multiply. Anything
  else (a bad key, a 404 model) fails immediately, because retrying can't fix it. So does a
  *daily* quota 429 (`_is_daily_quota`): it resets in hours, not seconds.
- **`thinking_level="low"`.** Gemini 3 models "think" before answering, and the default level
  is high. For writing a spec or a test file, low is plenty and is many times faster.
- **`timeout=120`.** Without a request timeout, a stuck call waits forever. With one, it fails and
  you see why.
- **Strip code fences.** Models often wrap code in ```` ```python ```` even when told not to.
  `strip_code_fences` keeps just the code, so the file you write to disk is valid Python.

**Check:** run both agents on a sample task, read the spec and tests, then run the tests against
a solution you know is correct.

```python
from devteam.agents import architect, test_writer
from devteam.tools import run_tests

state = {"task": "is_palindrome ignoring case and punctuation"}
state |= architect(state)      # merge the partial update, like LangGraph will
state |= test_writer(state)

print(state["spec"])
print(state["tests"])

reference = '''def is_palindrome(text: str) -> bool:
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    s = [c.lower() for c in text if c.isalnum()]
    return s == s[::-1]
'''
result = run_tests(reference, state["tests"])
print(result.passed, result.score)
print(result.output)
```

Expected: a spec of about 30 lines with all five sections, and a test file with a few
parametrised test functions (often 15–25 test cases in total), each call taking a few seconds.
Against the reference, either every test passes, or the failures point to **wrong tests**.
Both outcomes are useful: see Gotchas.

**Try it**
1. Run the check twice. Compare the two specs: what changed, and would the tests differ?
2. Remove the `## Errors` section from `ARCHITECT_SYSTEM` and rerun. Do the tests still check
   `TypeError`?
3. Change `RETRY_WAITS` to `[1, 1, 1]` and run the check several times quickly in a row to provoke a
   429. Watch the `[retry]` lines.
4. Set `MODEL_NAME` to a model that doesn't exist. Notice it fails at once, without retrying.

**Check yourself**
1. Why does the test writer get the spec but not the task?
2. Why turn off the library's built-in retries?
3. The tests fail against a correct reference solution. Whose fault is it, and which state
   field and route will handle this later?
4. What does `state |= architect(state)` imitate?

<details><summary>Answers</summary>

1. The spec is the single contract. If the test writer re-read the task, it could interpret it
   differently from the coder and the two would drift apart.
2. Two retry layers multiply the waiting time and hide what's happening. One visible layer is
   easier to reason about.
3. The tests are wrong. In step 6 the debugger sets `blame="test"`. In step 8 that sends the graph
   back to the test writer, once only (`test_fix_used`).
4. LangGraph merging a node's partial update into the state (the default "replace" reducer).
</details>

**Gotchas**
- **LLM tests can be wrong.** In one run the test writer asserted that
  `"Ana, ¿habla la bahía? ¡Ana!"` is a palindrome. It isn't: `í` and `i` are different characters,
  so the cleaned string doesn't read the same backwards. Correct code would fail that test. This is
  exactly why the debugger can blame the tests, and why you should check generated tests against a
  known-good solution while developing.
- **Specs and tests vary between runs.** Temperature 0.2 reduces this but doesn't remove it.
  Don't compare runs line by line; compare whether they're *correct*.
- **A slow or overloaded model.** Free-tier capacity varies per model and per hour. If a call
  takes minutes or returns `504 DEADLINE_EXCEEDED` / `503 UNAVAILABLE`, time a one-word prompt on
  a few Flash models and pick one that answers in seconds:
  ```python
  import time
  from dotenv import load_dotenv; load_dotenv()
  from langchain_google_genai import ChatGoogleGenerativeAI
  for m in ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash"]:
      t = time.time()
      try:
          ChatGoogleGenerativeAI(model=m, max_retries=0, timeout=60).invoke("Reply with: OK")
          print(m, f"{time.time() - t:.1f}s")
      except Exception as e:
          print(m, "error:", type(e).__name__)
  ```
- **`.env` must be `NAME=value`.** A key pasted on its own line, without `GOOGLE_API_KEY=` in
  front, is silently ignored and you get *"No API key was provided"*. Never print the parsed
  `.env` to debug this: a pasted-alone key is parsed as a variable *name*, so printing the names
  would leak it.
- **Windows console encoding.** Printing a spec with characters like `¿` or `日` can raise
  `UnicodeEncodeError` on a Windows console. Run Python in UTF-8 mode: `python -X utf8 ...` or
  set `PYTHONUTF8=1`.
- **Don't name helper functions `test_*` in test files.** pytest collects them. (Our `test_writer`
  lives in `agents.py`, which pytest never collects, so it's safe.)

---

### Step 4: Coder, standalone

**Goal:** an LLM function that turns a spec into `solution.py`, checked with the step 2 tools.

**Concepts:** [Role boundaries](#116-role-boundaries), [Ground truth](#14-ground-truth),
[Context engineering](#113-context-engineering-lean-prompts).

**Code:** append the coder prompts to `devteam/prompts.py`:

````python
CODER_SYSTEM = """\
You are a Python developer. Implement the spec you are given as a single module, `solution.py`.

Rules:
- Output ONLY the Python code of `solution.py`, no explanations.
- Implement exactly the spec: same function names, signatures and exception types.
- Standard library only. No `print`, no `input()`, no code at module level besides imports,
  constants and definitions.
- You never see or change the tests; the spec is your only contract.

The code must also pass a strict linter (ruff). In particular:
- Imports sorted (stdlib `import x` lines, then `from x import y`), none unused.
- Modern type hints: `list[int]`, `dict[str, int]`, `X | None` (not `List`, `Optional`).
- No unused variables, no bare `except:`, no `except Exception:`.
- When rejecting a value because of its type (after `isinstance`), raise `TypeError`.
- Don't end a function with a bare `return None`; prefer simple, direct code.
"""

CODER_USER = "Spec:\n\n{spec}"

CODER_RETRY = """

Your previous `solution.py` failed the checks:

```python
{code}
```

Diagnosis:
{diagnosis}

Last test output (may be truncated):
{test_output}

Lint output:
{lint_output}

Write a corrected `solution.py` that fixes these problems and still follows the spec.
"""
````

In `devteam/agents.py`, add one constant under `RETRY_WAITS`:

```python
MAX_OUTPUT_CHARS = 3000  # keep only the tail of test output in prompts (lean context)
```

and append the coder node at the end of the file:

```python
def _tail(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Last `limit` characters of text; pytest's summary is at the end."""
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


def coder(state: DevState) -> dict:
    """spec (+ previous code, diagnosis, check output on retries) -> solution.py source."""
    iteration = state.get("iteration", 0) + 1
    is_retry = bool(state.get("code"))
    print(f"[coder] iteration={iteration} retry={is_retry}")

    user = prompts.CODER_USER.format(spec=state["spec"])
    if is_retry:
        user += prompts.CODER_RETRY.format(
            code=state["code"],
            diagnosis=state.get("diagnosis") or "(none)",
            test_output=_tail(state.get("test_output") or "(none)"),
            lint_output=state.get("lint_output") or "(none)",
        )

    code = strip_code_fences(call_llm(prompts.CODER_SYSTEM, user))
    print(f"[coder] solution.py: {len(code.splitlines())} lines\n{_preview(code)}")
    return {"code": code, "iteration": iteration}
```

**Design notes**
- **The coder never sees the tests.** Its only contract is the spec. If it could read the tests,
  it would be tempted to code *to the tests* (special-casing an expected value) instead of to the
  behaviour. On a retry it sees the test *output*, which shows what failed but not the whole test file.
- **Same node, two modes.** If the state has no `code` yet, this is a first attempt: spec only.
  Otherwise it's a retry: the spec, plus the previous code, the diagnosis, and the test and lint
  output. That retry input is exactly what step 6's loop will feed back.
- **Lean retries.** `_tail` keeps only the last 3,000 characters of the test output. That's
  where pytest puts the failures and the summary line. The full history of earlier attempts is
  never sent.
- **The coder owns `iteration`.** Each call is one attempt, so the coder increments the counter.
  The cap in step 7 reads it.
- **Lint rules belong in the prompt.** Recent ruff versions enable a large default rule set:
  sorted imports, modern type hints, simplifications, and even "raise `TypeError` for a failed
  `isinstance` check". Code that passes every test can still fail on lint. Naming the rules most
  likely to trip in the prompt lets the coder pass both checks on the first attempt.

**Check:** run the three agents on a task, then check the coder's code with the step 2 tools.

```python
from devteam.agents import architect, coder, test_writer
from devteam.tools import run_lint, run_tests

state = {"task": "Merge overlapping intervals"}
state |= architect(state)
state |= test_writer(state)
state |= coder(state)

tests = run_tests(state["code"], state["tests"])
lint = run_lint(state["code"])
print("tests:", tests.passed, tests.score, "| lint clean:", lint.clean)
print(tests.output[-1500:])
```

Expected: `iteration == 1`, lint clean, and usually every test passing. If a test fails, read the
failure before you blame the coder (see Gotchas). Try tasks 1–4 from
[Sample tasks](#sample-tasks). Task 5 is meant to need fix rounds, so leave it for step 6.

**Try it**
1. **Exercise the retry path by hand.** Plant a bug in a passing solution and give the coder the
   failure:
   ```python
   bad = state["code"].replace("<=", "<", 1)       # e.g. touching intervals no longer merge
   r, l = run_tests(bad, state["tests"]), run_lint(bad)
   print(r.score, r.output.splitlines()[-1])
   retry = {**state, "code": bad, "test_output": r.output, "lint_output": l.output,
            "diagnosis": "(no debugger yet - read the test output)"}
   retry |= coder(retry)                            # prints retry=True, iteration=2
   print(run_tests(retry["code"], state["tests"]).score)
   ```
2. Delete the lint rules from `CODER_SYSTEM` and run tasks 1–4 again. How many lint failures?
3. Let the coder see the tests: add `state["tests"]` to the user message. Does it start
   special-casing values from the tests?

**Check yourself**
1. Why doesn't the coder get the test file?
2. How does `coder` know whether it's a first attempt or a retry?
3. Why send only the tail of the test output?
4. All tests pass but `run_lint` says not clean. Is the attempt a success?

<details><summary>Answers</summary>

1. So it implements the spec, not the particular asserts. Tests are a check, not a target.
2. `state.get("code")` is empty on the first attempt and filled on every later one.
3. pytest prints failures and the summary at the end. The rest costs tokens and buries the
   signal.
4. No. `passed` (step 5) requires tests *and* lint to be clean, so the loop will send it back.
</details>

**Gotchas**
- **Failing tests aren't always the coder's fault.** In practice the test writer sometimes invents
  non-ASCII examples and gets them wrong. For example it asserted `"Ésépé"` is a palindrome, but
  lowercased it's `ésépé`, which reads `épésé` backwards. The coder's code was correct. When a single test fails, check its
  expected value by hand first. This is the case the `blame="test"` route (step 8) exists for.
- **The coder may still add a `print` or an `if __name__ == "__main__":` demo.** It's harmless
  for the tests, but the prompt asks for no module-level code, to keep the output predictable.
- **An empty or truncated reply.** If `solution.py` comes back with only a few lines, the request
  may have hit the free-tier output limit or the timeout. Rerun, or look at the raw reply.

---

### Step 5: Linear graph

**Goal:** wire architect → test_writer → coder → run_checks → END into a real LangGraph graph.

**Concepts:** [Graph, node, edge](#15-graph-node-edge), [State](#16-state),
[Partial update and reducer](#17-partial-update-and-reducer).

**Check:** one full pass from task to test results, with each node printing its name and key fields.

_(Full walkthrough with code coming in the next revision.)_

---

### Step 6: Debugger + conditional edges

**Goal:** add the debugger and the first loop.

**Concepts:** [Structured output](#112-structured-output),
[Conditional edge](#18-conditional-edge-routing), [Cycle](#19-cycle-loop).

**Check:** a hard task (the CSV parser) goes round the loop at least once.

_(Full walkthrough with code coming in the next revision.)_

---

### Step 7: Cap, best attempt, run logs

**Goal:** guarantee the loop stops and always return the best code produced.

**Concepts:** [Termination guarantee](#110-termination-guarantee),
[Best-attempt tracking](#111-best-attempt-tracking).

**Check:** with the cap set to 1, the run stops with `status="max_iterations"` and returns `best_code`.

_(Full walkthrough with code coming in the next revision.)_

---

### Step 8: Test-blame route

**Goal:** let the debugger send work back to the test writer once, when the test itself is wrong.

**Concepts:** [Role boundaries](#116-role-boundaries),
[Conditional edge](#18-conditional-edge-routing).

**Check:** a deliberately wrong test is rerouted to the test writer exactly once.

_(Full walkthrough with code coming in the next revision.)_

---

### Step 9: CLI

**Goal:** `python main.py "task"` runs the whole system and reports the result.

**Check:** run all 5 sample tasks and record iterations and pass rate for each.

_(Full walkthrough with code coming in the next revision.)_

---

### Step 10 (optional): Tracing with MLflow

**Goal:** see every node, prompt and response for every run in a UI, and compare runs.

**Check:** `mlflow ui` shows a trace per run, plus per-task metrics (`status`, `iteration`, `best_score`).

_(Full walkthrough with code coming in the next revision.)_

---

### Sample tasks

Use these to exercise the system, roughly easiest to hardest:

1. `is_palindrome`, ignoring case and punctuation
2. Roman numerals, both directions
3. Parse durations like `"1h30m15s"` into seconds
4. Merge overlapping intervals
5. A CSV line parser with quoted fields and escaped quotes (usually needs several fix rounds)

---

## Part 4: Adapting the pattern to your own systems

The core of this project is a **generator–verifier loop**:

```
generate ──► verify (tool) ──► pass? ──► done
   ▲                              │ no
   └──────── diagnose ◄───────────┘
```

Swap the pieces and you get a different system:

| Domain | Generator | Verifier (ground truth) |
|---|---|---|
| SQL assistant | write a query | run it on a test DB, compare to expected rows |
| Data pipelines | write a transform | schema checks + row counts + sample assertions |
| Config / infra | write YAML or Terraform | `terraform validate`, schema validators, dry runs |
| API clients | write client code | call a mock server, check the responses |
| Structured extraction | extract fields from text | Pydantic validation + rule checks |
| Docs / content | write a section | link checker, spell check, required-sections checklist |

**A checklist for designing your own:**
1. What's the **objective check**? If there isn't one, a retry loop gives you much less.
2. What does each role **need to see**, and nothing more?
3. Which roles must **never** be allowed to change which artifacts?
4. How do you **score** partial success?
5. What's the **cap**, and what do you return when you hit it?
6. Which decisions need **structured output** so the router can branch on them?

**Where to go after v1:**
- Real isolation: run the checks in Docker.
- Multi-file projects: make `code` a dict of filename → contents.
- Persistence: LangGraph **checkpointers** save state between steps, so a run can resume or
  pause for a human to approve the next step ("human-in-the-loop").
- A different model per role: for example a cheap model for the architect and a stronger one for the coder.

---

## Part 5: FAQ

### Is LangGraph "loop-based"? Why is there no loop until step 6?
LangGraph is **graph-based**. A graph only loops if you add a conditional edge that points
backwards. With only fixed edges it's a straight pipeline. LangGraph *supports* cycles: state
carries over between rounds and a recursion limit protects you. The older LangChain chains could
only run in a straight line. We add the loop in step 6, once every piece has been tested on its own.

### CrewAI vs LangGraph: which should I use?
The difference is **where the loop lives and who controls it**.

| | CrewAI | LangGraph |
|---|---|---|
| Main idea | Agents with roles doing tasks | Nodes and edges over shared state |
| Process | Sequential or hierarchical (a manager agent hands out work) | Whatever graph you draw |
| Loops | **Inside each agent**, automatic: think → call tool → observe → repeat | **Between nodes**: you build them with conditional edges |
| Team-level retry ("tests failed, back to the coder") | Not by default; add it yourself (e.g. Flows with routers) | Just a conditional edge |
| Control | The framework decides more | You decide everything |
| Feels like | Managing a team | Drawing a flowchart |

**CrewAI is easier to start with. LangGraph gives you more control.** This project's core is a
custom loop with a cap and a one-time test fix, which is exactly the kind of control LangGraph is
built for.

### Why not let one LLM do everything in one prompt?
Long, multi-purpose prompts give weaker results and are hard to debug. Separate roles give
you focused prompts and show you *which* part failed.

### Why Gemini's free tier?
You can build and experiment for free. The trade-off is rate limits, which is why every call has
backoff and why prompts stay lean. Check the **daily** limit in AI Studio before a long session:
it can be as low as 20 requests per model per day, and one graph pass is at least 3 calls (plus 2
per fix round). Each model has its own quota, so switching `MODEL_NAME` gives you a fresh one.
Note that some model IDs are aliases: the error may name a different model than the one you asked
for, and those IDs share a quota. Because the model is created in one `get_llm()` function,
moving to a paid model or another provider is a one-line change.

### Is the temp-dir sandbox safe?
No, not for untrusted input. It protects against stale files and infinite loops, not against
malicious code. Only run tasks you wrote yourself, or use containers.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `429` / `RESOURCE_EXHAUSTED` | Free-tier rate limit | Backoff will retry. Wait, or switch to the fallback provider |
| *"Daily free-tier quota ... is used up"* | Per-day request cap hit (`GenerateRequestsPerDay...`) | Switch `MODEL_NAME` to another model (separate quota), use the fallback provider, or wait for the reset |
| Model not found | Model ID renamed or retired | Check the current IDs in Google AI Studio and update `MODEL_NAME` |
| Tests "pass" with 0 tests | Should be impossible here: exit code 5 counts as a failure | Check `run_tests` still compares `returncode == 0` |
| Every run says `TIMEOUT` | Generated code loops, or a test waits for input | Read `test_output`. The debugger should flag it |
| `GraphRecursionError` | Your cap is missing or above LangGraph's recursion limit | Check the `iteration` routing, or raise `recursion_limit` |
| `ModuleNotFoundError: solution` | Tests import a different module name | Tighten the test writer prompt: "import from `solution`" |
| *"No API key was provided"* | `.env` line isn't `GOOGLE_API_KEY=...` (e.g. key pasted alone) | Fix the line format. Don't print the parsed `.env` to debug it |
| A call takes minutes, or `504 DEADLINE_EXCEEDED` | The model is overloaded, or thinking at a high level | Use `thinking_level="low"`; time a one-word prompt on other Flash models and switch `MODEL_NAME` |
| `503 UNAVAILABLE` ("high demand") | Temporary model overload | Backoff retries it. If it keeps happening, switch model |
| `UnicodeEncodeError` when printing | Windows console isn't UTF-8 | Run `python -X utf8 ...` or set `PYTHONUTF8=1` |
| Correct code fails a generated test | The test is wrong (LLMs make mistakes too) | Expected now and then. The debugger's `blame="test"` route handles it from step 8 |
| All tests pass but lint fails (`I001`, `UP006`, `TRY004`, ...) | Recent ruff enables many rules by default | List the commonly tripped rules in `CODER_SYSTEM`. Retries get `lint_output` to fix the rest |

---

## Glossary

Quick definitions. See [Part 1](#part-1-core-concepts) for the full explanations.

| Term | Meaning |
|---|---|
| **Agent** | A function that calls an LLM with a focused role. |
| **Multi-agent system** | Several narrow agents passing work between them. |
| **LLM node** | A graph node that calls a model. |
| **Tool node** | A graph node that runs ordinary code (no LLM), like `run_checks`. |
| **Ground truth** | An objective correctness signal, such as test results. |
| **State** | The shared dict all nodes read and update. Ours is `DevState`. |
| **Node** | A function `state -> partial update`. |
| **Edge** | A fixed "after A, run B" connection. |
| **Conditional edge** | An edge whose target is chosen at runtime by a routing function. |
| **Cycle** | A path back to an earlier node. This is how LangGraph loops. |
| **Partial update** | The dict a node returns, containing only the fields it changed. |
| **Reducer** | The rule for merging an update into a field. The default is "replace". |
| **`START` / `END`** | Special markers for where a graph begins and stops. |
| **Compile** | `builder.compile()` turns a graph definition into something you can run. |
| **`invoke` / `stream`** | Run a compiled graph to completion, or yield the update from each node as it runs. |
| **Recursion limit** | LangGraph's built-in cap on graph steps (default 25). |
| **Structured output** | LLM output forced into a Pydantic schema. |
| **Backoff** | Waiting longer and longer between retries after rate-limit errors. |
| **Transient error** | A temporary failure (429 quota, 503 overloaded) that a retry can fix. A 401 or 404 can't be fixed by retrying. |
| **System message** | The fixed part of a prompt: the agent's role and rules. |
| **User message** | The per-run part of a prompt: this task, this spec. |
| **Spec** | The architect's written contract that both the tests and the code are built from. |
| **Thinking level** | How much hidden reasoning a Gemini 3 model does before answering. Lower is faster. |
| **Retry input** | What the coder gets on a later attempt: spec + previous code + diagnosis + the tail of the check output. |
| **Coding to the tests** | Special-casing expected values instead of implementing the behaviour. This is why the coder never sees the test file. |
| **Code fence** | The ```` ``` ```` markers models wrap code in. They must be stripped before writing a `.py` file. |
| **Best-attempt tracking** | Keeping the highest-scoring result in case the run never fully succeeds. |
| **Checkpointer** | A LangGraph component that saves state between steps (resume, human-in-the-loop). |
| **Agent loop (ReAct)** | The think → act → observe loop *inside* one agent. CrewAI does this automatically. |
| **CrewAI** | A role-based multi-agent framework: agents + tasks + a process. Less wiring, less control. |
