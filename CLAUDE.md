# DevTeam — LangGraph multi-agent coding loop

## Goal
A LangGraph multi-agent "dev team" that takes a small Python task, writes tests,
writes code, runs the tests for real, and loops on failures until they pass or a cap is hit.
I am building this myself to learn LangGraph. Build in small steps, explain each piece
before implementing it, and stop after each step so I can run and check it.

## Scope (v1)
- Single-file Python tasks only (`solution.py` + `test_solution.py`)
- CLI only, no frontend: `python main.py "task description"`
- One LLM model for all agents
- Python 3.11+ (use 3.14; 3.15 lacks a zstandard wheel), venv, deps in requirements.txt:
  langgraph, langchain-google-genai, pytest, ruff, python-dotenv (mlflow optional, step 10)

## LLM provider: Google Gemini (free tier)
- Use `ChatGoogleGenerativeAI` from `langchain-google-genai`; key is `GOOGLE_API_KEY` in `.env`
- Model: a free-tier Flash model (e.g. `gemini-3.8-flash`); set it ONCE as `MODEL_NAME` in
  `devteam/agents.py` so it's easy to change. Check exact model IDs in Google AI Studio.
- Free tier has per-minute and per-day limits: wrap every LLM call with retry + backoff on
  429 / RESOURCE_EXHAUSTED (e.g. wait 10s, 20s, 40s, then fail clearly)
- Keep prompts lean (send only current code, tests and last test output, not full history)
- Fallback provider if limits hit: Groq via `langchain-groq` (`GROQ_API_KEY`); keep the model
  creation in one `get_llm()` function so switching is a one-line change

## Roles
LLM agents:
- **architect**: task -> spec (function signatures, behaviour, edge cases, examples). No code.
- **test_writer**: spec -> one pytest file importing from `solution`.
- **coder**: spec (+ diagnosis + test output on retries) -> `solution.py` only. Must NEVER edit tests.
- **debugger**: structured output (Pydantic): `diagnosis: str`, `blame: Literal["code","test"]`, `fix_hint: str`.

Tool nodes (no LLM):
- **run_checks**: writes files to a temp dir, runs `pytest -q --tb=short` (30s timeout) and `ruff check solution.py`.

## State (TypedDict)
task, spec, tests, code, test_output, lint_output, passed, diagnosis, blame,
iteration, test_fix_used, best_code, best_score, status

## Flow
architect -> test_writer -> coder -> run_checks
- run_checks: passed (tests AND lint clean) -> END, status="success"
- run_checks: iteration >= 5 -> END, status="max_iterations", output best_code
- run_checks: otherwise -> debugger
- debugger: blame=="test" and not test_fix_used -> test_writer (set test_fix_used=True)
- debugger: otherwise -> coder
- coder increments `iteration`

## Sandbox rules
- New temp dir per check run; subprocess with timeout; timeout counts as failure ("TIMEOUT" in output)
- Score = number of passed tests (parse pytest summary line); keep best_code / best_score
- Not real isolation: only run tasks I wrote myself

## Project layout
```
main.py            # CLI entry
runs/              # per-run folder, per-iteration code + diagnosis logs (gitignored)
docs/
  README.md        # index of all docs (existing + planned)
  LEARNING.md      # learning log, one section per build step + glossary
  setup.md         # environment setup + how to run each step's check
  tasks.md         # test tasks
devteam/
  state.py
  prompts.py
  agents.py        # architect, test_writer, coder, debugger
  tools.py         # run_tests, run_lint
  graph.py
```
Gitignore: .env, runs/, mlruns/, .venv/

## Build order (commit after each)
1. Setup + state. Check: imports cleanly.
2. tools.py. Check: good code passes, bad code fails, infinite loop times out.
3. architect + test_writer standalone. Check: read outputs.
4. coder. Check: its code passes tests when run manually.
5. Linear graph (no loop). Check: one full pass.
6. Debugger + conditional edges. Check: task 5 iterates.
7. Cap + best attempt + runs/ logging. Check: cap=1 forces stop.
8. Test-blame route. Check: a planted wrong test is rerouted once.
9. CLI. Check: all 5 tasks run; note iterations and pass rate.
10. (Optional) MLflow tracing: `mlflow.set_experiment("devteam")`, `mlflow.langchain.autolog()`, view with `mlflow ui`.

## Test tasks
1. is_palindrome ignoring case and punctuation
2. Roman numerals both directions
3. Parse durations like "1h30m15s" to seconds
4. Merge overlapping intervals
5. CSV line parser with quoted fields and escaped quotes (should need fix rounds)

## Conventions
- Print node name + key state fields at each node while developing
- Keep prompts in prompts.py, not inline
- Don't add features beyond the current step
- After each build step, update docs/LEARNING.md before committing: replace that step's placeholder
  with a full walkthrough (Goal, Concepts, Code, Design notes, Check, Try it, Check yourself,
  Gotchas) and add new concepts/glossary/FAQ entries. It's a sellable guide for other developers:
  self-contained (code inline), no personal notes
- New docs go in docs/ and get a line in docs/README.md
