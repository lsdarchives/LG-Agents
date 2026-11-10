# DevTeam — LangGraph multi-agent coding loop

> **v1 (below) is finished.** v2, an autonomous team of agents defined in `agents.yaml`, has its spec,
> build order (steps 11+) and conventions in [CLAUDE_v2.md](CLAUDE_v2.md); follow it for all new work.
> The v2 guide is docs/06-v2-guide.md (updated after each v2 step, like LEARNING.md for v1).

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
  langgraph, langchain-google-genai, langchain-groq, pytest, ruff, python-dotenv (mlflow optional, step 10)

## LLM provider: Groq (free tier), Gemini as fallback
- Primary: `ChatGroq` from `langchain-groq`, model `openai/gpt-oss-120b`, `reasoning_effort="low"`;
  key is `GROQ_API_KEY` in `.env`. Free tier is far larger than Gemini's (reported ~1K req/day)
- Fallback: `ChatGoogleGenerativeAI` (`gemini-3.5-flash`, `thinking_level="low"`), `GOOGLE_API_KEY`
- Set `PROVIDER` ONCE in `devteam/agents.py` (`MODEL_NAMES` holds one model per provider); all
  model creation stays in `get_llm()` so switching is a one-line change
- Free tier has per-minute and per-day limits: wrap every LLM call with retry + backoff on
  429 / RESOURCE_EXHAUSTED and 503 / UNAVAILABLE (wait 10s, 20s, 40s, then fail clearly);
  library retries off (`max_retries=0`) so they don't stack. A per-day quota 429 fails at once
  (Gemini free tier can be 20 requests/model/day; waiting seconds won't help)
- Keep prompts lean (send only current code, tests and last test output, not full history)

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
  README.md        # index of all docs, in reading order
  01-setup.md      # environment setup + how to run each step's check
  LEARNING.md      # (02) the guide; plain name so it can be shared standalone
  03-tasks.md      # test tasks
  04-results.md    # stub until step 9: iterations + pass rate per task
  05-tracing.md    # stub until step 10 (optional): MLflow tracing
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
- After each build step, also add that step's check to docs/01-setup.md and update the Status
  column in docs/README.md (fill 04-results.md at step 9, 05-tracing.md at step 10)
- New docs go in docs/ with the next number prefix (NN-name.md) and get a row in docs/README.md
