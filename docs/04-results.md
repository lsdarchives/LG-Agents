# Results

Every sample task in [03-tasks.md](03-tasks.md) was run once through the full graph with
`python main.py "<task>"`, and the outcome is recorded here.

## What is recorded

| Column | Meaning |
|---|---|
| Task | Which sample task (1–5) |
| Status | `success` or `max_iterations` |
| Attempts | How many coder attempts it took (`iteration` at the end), out of the cap of 5 |
| Score | Passed tests for the final / best code (`best_score`) |
| Test rewrite | Whether the debugger blamed the tests and they were rewritten (`test_fix_used`) |
| Time | Wall-clock seconds for the whole run, including rate-limit waits |

## Run: 2026-11-09 (v1)

Setup: Groq `openai/gpt-oss-120b`, `reasoning_effort="low"`, `MAX_ITERATIONS = 5`, one run per task.
LLM calls weren't counted for this run.

| # | Task | Status | Attempts | Score | Test rewrite | Time |
|---|---|---|---|---|---|---|
| 1 | is_palindrome | success | 2 | 24 | yes | 31s |
| 2 | Roman numerals | success | 1 | 29 | no | 55s |
| 3 | Durations | success | 3 | 20 | yes | 115s |
| 4 | Merge intervals | max_iterations | 5 | 23 (all pass) | yes | 620s |
| 5 | CSV line parser | success | 4 | 27 | yes | 199s |

**Pass rate: 4/5** (5/5 if you count tests alone).

### Notes

- **The test rewrite fired in 4 of 5 tasks, each time for a genuinely wrong test.** In task 1 a
  `parametrize` call declared 3 argument names but supplied only 2 values. In task 3 a test expected
  an error for input the spec defined as valid. In task 5 a test's expected whitespace contradicted
  the spec. After the rewrite, the debugger correctly went back to blaming the code.
- **Task 4 failed only on lint.** After the rewrite, every attempt passed all 23 tests. But the
  coder regenerates the whole file each time, and each version brought in a different unused import
  (`itertools`, then `builtins`, then `typing.Any`), so ruff never came back clean. Task 5 also lost
  one attempt to an unused `import sys`.
- **Rate limits:** several 429s across the runs. Every one recovered on the first 10s wait.
- **Known improvement (not applied in v1):** run `ruff check --fix` on `solution.py` before linting.
  That removes unused imports and sorts the rest automatically, so lint-only failures like task 4's
  would disappear without spending coder attempts.
