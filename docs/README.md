# Docs

Project spec and build order live in [../CLAUDE.md](../CLAUDE.md), which stays at the repo root so
Claude Code loads it automatically. Everything else is here.

## Reading order

| # | Doc | What it's for | Status |
|---|---|---|---|
| 01 | [01-setup.md](01-setup.md) | Set up the environment from scratch and run each step's check | Done; a check is added per step |
| 02 | [LEARNING.md](LEARNING.md) | The guide, written for other developers: core concepts, architecture, a walkthrough per build step, how to adapt the pattern, FAQ, troubleshooting, glossary | Steps 1-10 written; updated after each step |
| 03 | [03-tasks.md](03-tasks.md) | The 5 test tasks the dev team is evaluated on | Done |
| 04 | [04-results.md](04-results.md) | Iterations and pass rate per test task | Done (v1 run 2026-11-09) |
| 05 | [05-tracing.md](05-tracing.md) | MLflow setup and how to read the traces | Done (optional step 10) |
| 06 | [06-v2-guide.md](06-v2-guide.md) | The v2 guide: from a coding loop to an autonomous team (agent registry, supervisor, tools, artifacts), a walkthrough per step from 11 | Steps 11-12 written; updated after each v2 step |

`LEARNING.md` keeps a plain, unnumbered name so it can be shared on its own as a standalone guide.

New docs get the next number and a row here.
