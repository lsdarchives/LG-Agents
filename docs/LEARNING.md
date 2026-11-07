# Learning LangGraph by building DevTeam

A learning log that grows with the project. Each build step gets a section explaining
what we built, why, and how to poke at it yourself.

## How to use this file

1. Read a step's section **before** running its check.
2. Do the **Try it** exercises (run them from the repo root with the venv active).
3. Answer **Check yourself** without looking, then open the answers.
4. Write anything that surprised you in **My notes** at the bottom.

Where things live: [docs/README.md](README.md) indexes every doc; setup steps are in [setup.md](setup.md).

---

## Big picture

```
            ┌──────────────────────────────────────────────┐
            │                                              │
task ──► architect ──► test_writer ──► coder ──► run_checks ──► END (success / max_iterations)
                          ▲              ▲          │
                          │              │          ▼
                          └── blame=test ┴─ debugger
                             (once only)    blame=code
```

Core LangGraph ideas, and the step where each one first appears:

| Idea | One line | Step |
|---|---|---|
| **State** | One shared dict (here a `TypedDict`) that every node reads from and writes to | 1 |
| **Node** | A plain Python function `state -> partial update dict` | 3–5 |
| **Edge** | A fixed "after A, run B" connection | 5 |
| **Conditional edge** | A function that looks at the state and returns the name of the next node | 6 |
| **Cycle + END** | Edges can loop back; `END` stops the graph. Something must guarantee the loop ends | 6–7 |

The LLM is just *one kind* of node. LangGraph doesn't care whether a node calls Gemini,
runs pytest or just does arithmetic, as long as it takes state and returns an update.

---

## Step 1: Setup + state

### Concept
**State is the graph's only memory.** Nodes don't call each other or pass arguments.
Each node gets the current state, returns a dict of *only the fields it changed*, and LangGraph
merges that into the state before running the next node.

### What we built
- [devteam/state.py](../devteam/state.py): `DevState`, a `TypedDict` with every field the graph uses.
- [requirements.txt](../requirements.txt), [.gitignore](../.gitignore), venv, `.env` with `GOOGLE_API_KEY`.

### Why it's designed this way
- **`TypedDict`, not a class or Pydantic model:** it's the default LangGraph state type, and it's just a
  dict at runtime, so nodes can return `{"code": "..."}` with no ceremony. The type hints are
  for you and your editor. **Python does not check them at runtime.**
- **Merge rule = last write wins:** with no extra annotation, each field uses LangGraph's default
  behaviour. A node that returns `{"code": new}` *replaces* `code`. (You can attach a *reducer*,
  e.g. to append to a list instead, but we don't need one in v1.)
- **The fields are grouped by who writes them:**

  | Group | Fields | Written by |
  |---|---|---|
  | Input | `task` | CLI |
  | Artifacts | `spec`, `tests`, `code` | architect, test_writer, coder |
  | Check results | `test_output`, `lint_output`, `passed` | run_checks |
  | Debugger verdict | `diagnosis`, `blame` | debugger |
  | Loop control | `iteration`, `test_fix_used` | coder, debugger route |
  | Best attempt | `best_code`, `best_score` | run_checks |
  | Outcome | `status` | the end of the graph |

- **The loop-control fields exist because graphs can cycle.** Without `iteration`, a failing task would
  loop forever. Without `test_fix_used`, the debugger could keep blaming the tests.

### Try it
```python
from typing import get_type_hints
from devteam.state import DevState

print(get_type_hints(DevState))          # every field and its type

s: DevState = {"task": 123}               # wrong type, missing fields...
print(s)                                  # ...runs anyway: TypedDict is not validated
```

### Check yourself
1. A node returns `{"code": "x = 1"}`. What happens to `tests` in the state?
2. Why is `blame` allowed to be `None`?
3. Which field stops the graph from looping forever?

<details><summary>Answers</summary>

1. Nothing. It keeps its previous value. Only the fields a node returns are updated.
2. Before the debugger has run for the first time there is no verdict yet.
3. `iteration` (checked against the cap in run_checks' routing). `test_fix_used` limits the
   test-rewrite route to one use.
</details>

### Gotchas
- Typos in returned keys (`{"cod": ...}`) are a silent bug risk, so watch the printed state.
- **The newest Python isn't always installable.** On Python 3.15, `pip install` failed:
  `zstandard` (pulled in by langgraph's dependencies) had no prebuilt wheel yet, and building
  it from source needs the Microsoft C++ Build Tools. Recreating the venv with `py -3.14`
  fixed it. If you see *"Microsoft Visual C++ 14.0 or greater is required"*, check your Python version first.

---

## Step 2: tools.py (run tests + lint for real)

### Concept
**Not every node should be an LLM.** Checking code is deterministic: run pytest, read the
result. Doing it with real tools gives the loop *ground truth*. The LLM can't talk its way past
a failing test.

### What we built
[devteam/tools.py](../devteam/tools.py), plain functions (they become the `run_checks` node later):
- `run_tests(code, tests) -> TestResult(output, passed, score)`
- `run_lint(code) -> LintResult(output, clean)`
- `count_passed(output)`: parses `"3 failed, 5 passed in 0.12s"` → `5`
- `_run(cmd, cwd)`: shared subprocess wrapper with the 30s timeout

### Why it's designed this way
- **A fresh temp dir per run:** no leftover files from earlier attempts can affect a result.
  `TemporaryDirectory` deletes it afterwards.
- **`sys.executable -m pytest`:** runs pytest from *this venv's* Python, not whatever `pytest`
  is first on your PATH.
- **The timeout counts as a failure:** the LLM may write `while True:`. `subprocess.run(timeout=30)`
  kills it and we return `"TIMEOUT: ..."`, `passed=False`, `score=0`.
- **Exit codes:** pytest returns `0` only if all tests pass. `5` means *no tests collected*,
  e.g. an empty or broken test file. That's treated as a failure too, never as "nothing failed so it passed".
- **Score = number passed:** this lets us keep the *best* attempt even if we never reach 100% (step 7).
- **`ruff check --isolated`:** ignores any ruff config in parent folders, so lint results
  depend only on the generated code.
- **`NamedTuple` results:** this is the same idea as state. Return a small, named record, not a loose tuple.
- **Not a sandbox:** generated code runs as *you*. Only run tasks you wrote yourself.

### Try it
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

### Check yourself
1. Why use a new temp dir every run instead of one fixed `work/` folder?
2. An LLM writes a test file with a syntax error. What do `passed` and `score` come back as?
3. Why parse the score from the output when the exit code already says pass or fail?

<details><summary>Answers</summary>

1. Isolation between attempts: no stale `solution.py`, `__pycache__` or cache files can change
   the outcome.
2. `passed=False` (collection error gives a non-zero exit code), `score=0` (no "N passed" in the output).
3. The exit code is all-or-nothing. The score shows *partial progress*, which we need to
   pick the best attempt when the cap is hit.
</details>

### Gotchas
- On Windows, temp-dir cleanup can fail if a process still holds a file open, which is why we pass
  `ignore_cleanup_errors=True`.
- `-p no:cacheprovider` / `--no-cache` stop pytest and ruff from writing cache folders.

---

## Step 3: architect + test_writer standalone
_Coming up: chat models in LangChain, prompts kept in `prompts.py`, retry/backoff on 429s._

## Step 4: coder
_Coming up: feeding only the context needed (lean prompts), checking LLM output by hand._

## Step 5: linear graph
_Coming up: `StateGraph`, `add_node`, `add_edge`, `START`/`END`, `compile()`, `invoke` vs `stream`._

## Step 6: debugger + conditional edges
_Coming up: structured output with Pydantic, `add_conditional_edges`, routing functions._

## Step 7: cap + best attempt + runs/ logging
_Coming up: guaranteeing loop termination, tracking the best result, per-run logs._

## Step 8: test-blame route
_Coming up: a second loop target, one-shot flags in state._

## Step 9: CLI
_Coming up: `main.py`, running all 5 tasks, reading the results._

## Step 10 (optional): MLflow tracing
_Coming up: autolog traces, metrics per run, `mlflow ui`._

---

## Side questions

### Is LangGraph "loop-based"? Why is there no loop until step 6?
LangGraph is **graph-based**. A graph only loops if you add a **conditional edge that points
backwards**. With only fixed edges it's a straight pipeline. What LangGraph adds is *support* for
cycles: state carries over between rounds, and a recursion limit stops runaway loops. The older
LangChain chains could only run in a straight line.

We add the loop late on purpose. Steps 3–4 test each agent on its own, step 5 wires them in a
straight line, and step 6 adds the loop. By then the loop is the only new thing, so when
something breaks you know where to look.

### CrewAI vs LangGraph: which one is loop-based?
Neither, really. The difference is **where the loop lives and who controls it**.

| | CrewAI | LangGraph |
|---|---|---|
| Main idea | Agents with roles doing tasks | Nodes and edges over shared state |
| Process | Sequential or hierarchical (a manager agent hands out work) | Whatever graph you draw |
| Loops | **Inside each agent**, automatic: think → call tool → observe → repeat | **Between nodes**: you build them with conditional edges |
| Team-level retry ("tests failed, back to the coder") | Not by default; add it yourself (e.g. Flows with routers) | Just a conditional edge |
| Control | The framework decides more | You decide everything |
| Feels like | Managing a team | Drawing a flowchart |

Short version: **CrewAI is easier to start with, LangGraph gives you more control.** DevTeam's
core is a custom loop (checks → debugger → coder, with a cap and a one-time test fix), which is
exactly the kind of control LangGraph is built for.

---

## Glossary

| Term | Meaning |
|---|---|
| **State** | The shared dict all nodes read and update. Ours is `DevState`. |
| **Partial update** | The dict a node returns, containing only the fields it changed. |
| **Reducer** | The rule for merging an update into a state field. The default is "replace". |
| **Tool node** | A node that runs ordinary code (no LLM), like `run_checks`. |
| **Cycle** | A path in the graph that leads back to an earlier node. This is how LangGraph loops. |
| **Conditional edge** | An edge whose target is chosen at runtime by a function that reads the state. |
| **Agent loop (ReAct)** | The think → act (tool call) → observe loop *inside* one agent. CrewAI does this automatically. |
| **CrewAI** | A role-based multi-agent framework: agents + tasks + a process. Less wiring, less control. |

---

## My notes

-
