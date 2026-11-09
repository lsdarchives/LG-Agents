# DevTeam v2 — autonomous multi-agent software team

> Paste this section under the v1 content in CLAUDE.md (or keep as docs/V2.md and reference it).
> v1 (single-file coder/tester loop) must be committed and tagged `v1` before starting.

## Goal
A team of 32 specialised agents, coordinated by a Tech Lead supervisor, that builds SMALL
multi-file projects in any supported stack: web and mobile apps, APIs, TypeScript apps, databases,
data engineering, ML/AI, deep learning, reinforcement learning, cloud infrastructure and UI design,
across domains (e-commerce, SaaS, services booking, data analysis, ML/AI, IDP) in a sandboxed `workspace/` folder. Agents are defined in `agents.yaml`,
stacks in `stacks.yaml`; neither is hard-coded.

## Stacks (stack-agnostic team)
- `stacks.yaml` holds stack profiles: language, Docker image, scaffold, install/test/lint/build/run/
  smoke commands, test framework and coding conventions. 44 profiles:
  - Web frontend: react-vite, vue-vite, angular, sveltekit, nextjs, nuxt, astro, react-router, static-web
  - Mobile: react-native-expo, flutter, kotlin-logic, swift-package
  - APIs/backend: fastapi, django, flask, express (js), nestjs, fastify, graphql (apollo, strawberry),
    go, rust-axum, spring-boot, dotnet, laravel, rails, python-cli
  - Data engineering: python-data-eng (Polars/DuckDB/Prefect), dbt-duckdb, pyspark-local
  - ML/AI: python-data-ml, python-ai-agents (LangGraph/RAG), python-pytorch, python-tensorflow, python-rl
  - Cloud/infra: terraform, aws-cdk-ts, pulumi-python, kubernetes-helm, github-actions, aws-sam-python
- `services` (run via docker compose): postgres, mysql, mongodb, redis, pgvector, chroma, kafka,
  minio, localstack. SQLite needs none.
- `presets` (popular combos): MERN, MEAN, MEVN, PERN, T3, LAMP-Laravel, Django-React, FastAPI-React,
  Spring-React, Rails-Hotwire, NestJS-Next, Next-Supabase-style, Mobile-Expo-API, Mobile-Flutter-API,
  RAG-App, Data-Platform, ML-Service, DL-Service, Cloud-Ready.
- The architect picks a profile PER COMPONENT (an app can be api=python-fastapi + web=react-vite-ts).
- The builder/test_engineer prompts get that component's profile injected; qa_runner and devops
  run the profile's commands. Adding a language = adding a profile.
- Honest limits (agents are told these):
  - Mobile: app logic + UI components tested headlessly; no device/emulator builds or store releases.
  - Cloud: write + validate/plan only (mocks, LocalStack); NEVER apply/deploy to real accounts.
  - Deep learning / RL: CPU-sized models, toy data, short runs; code scales up later on a GPU.
  - Design: specs, tokens and HTML mockups; no Figma files.
  - Unsupported ("blocked"): desktop GUI apps, game engines, embedded/firmware, GPU-scale training. I build this step by step: explain each piece before implementing it,
stop after each step so I can run the check, then commit.

## Design principles (from research — follow these)
1. **Structured artifacts, not chat.** Agents communicate through documents with fixed formats
   (PRD, domain notes, design, task board, test report, diagnosis, review), like MetaGPT's SOPs.
   Each agent only receives the artifacts listed in its `reads`.
2. **Execution feedback beats opinions.** Real pytest/ruff output drives every fix loop.
3. **Ask, don't guess.** Any agent may return `status: "blocked"` with questions instead of
   hallucinating missing details (ChatDev's "communicative dehallucination").
4. **Precise delegation.** Every Tech Lead handoff has: objective, inputs, expected output,
   boundaries (what NOT to touch). Vague delegation causes duplicated and missed work.
5. **Agent-friendly tools.** Small, guarded tools with clear feedback (SWE-agent's ACI findings):
   paged file viewer, edit command that rejects syntax errors, compact search.
6. **Scale effort to the task.** Only wake agents the project needs.
7. **Isolation.** Builders get a fresh context per task (task + design + relevant files only).
8. **Hard limits everywhere.** Autonomous routing can loop forever without caps.

## Architecture
```
Tech Lead (supervisor, strong model) ── handoff tool → Command(goto=agent)
 ├─ Planning:  product_manager → <domain>_advisor → architect → api_designer → ui_ux_designer
 │             → project_manager
 ├─ Build:     backend_dev | frontend_dev | mobile_dev | database_dev | integration_dev
 │             | data_engineer | ml_engineer | llm_engineer | deep_learning_engineer | rl_engineer
 │             | cloud_engineer | mlops_engineer
 ├─ Quality:   test_engineer → qa_runner (tool) → debugger → code_reviewer → security_reviewer
 │             → accessibility_reviewer
 └─ Ops:       devops → docs_writer
```
Per task loop: test_engineer → builder → qa_runner → (debugger → fix_owner → qa_runner)* →
next task. After all tasks: code_reviewer + security_reviewer → (fixes) → devops → docs → FINISH.

## Files
```
agents.yaml                 # agent registry (roles, prompts, tools, tiers, budgets)
stacks.yaml                 # stack profiles (commands, images, conventions)
devteam/
  config.py                 # model tiers -> model IDs, caps, concurrency
  llm.py                    # get_llm(tier) with retry/backoff on 429
  registry.py               # load + validate agents.yaml (pydantic)
  builder.py                # build_agent(spec) -> node (ReAct agent with tools) or tool node
  tools/workspace.py        # file tools, locked to workspace/
  tools/exec.py             # run_tests, run_command (allow-list), qa_runner
  artifacts.py              # read/write artifacts in workspace/.devteam/
  state.py                  # TeamState with reducers
  graph.py                  # supervisor graph, subgraphs, checkpointer
main.py                     # python main.py "build a ..." [--approve-plan] [--trace]
workspace/<run_id>/         # generated project + .devteam/ artifacts (gitignored)
```

## State (reducers)
```python
class TeamState(TypedDict):
    request: str
    messages: Annotated[list, add_messages]          # audit log only, not shared context
    artifacts: Annotated[dict[str, str], merge_dicts] # name -> path
    task_board: Annotated[list[Task], merge_tasks]    # by id; status todo|in_progress|done|failed
    current_task: str | None
    last_result: dict                                  # status, summary, questions, handoff_notes
    review_rounds: int
    steps_used: int
    status: str                                        # running|done|partial|failed
```

## Tools (agent-computer interface)
All paths resolved inside `workspace/<run_id>/`; anything escaping it is rejected.
- `list_tree(depth=3)`: compact file tree
- `view_file(path, start=1)`: 100 lines at a time, with line numbers, path and total lines
- `search_files(pattern)`: list of matching files only (+ match count), no big context dumps
- `create_file(path, content)`
- `edit_file(path, start_line, end_line, new_text)`: replace a line range; for .py files run a
  syntax check and REJECT the edit (file unchanged) if it fails; return the updated region
- `run_tests(target=None)`: runs the component's stack `test` command, timeout 120s, trimmed output
- `run_command(cmd)`: allow-list built from the stack profiles' commands (+ bandit, npm audit); timeout

## Sandbox
- Python-only components may run locally in a temp venv (as in v1).
- Every non-Python stack runs inside its profile's Docker image: workspace mounted at /work,
  `--network` enabled only for the install step, CPU/memory limits, container removed after.
- Requires Docker Desktop on Windows. Without Docker, only Python profiles are enabled.
- `write_artifact(name, content)` / `read_artifact(name)`
- Empty output always returns "Command ran successfully with no output."

## Limits (config.py)
- recursion_limit 200; max total agent activations 60
- max fix loops per task 4; max review rounds 2; max blocked→answer cycles 3
- per-agent `max_steps` from agents.yaml
- max parallel builders 2 (free-tier rate limits); token budget per run, stop gracefully
- Tech Lead must FINISH with status "partial" + reason when a cap is hit

## Models
Two tiers mapped in config.py: `strong` (tech_lead, architect, api_designer, ui_ux_designer,
test_engineer, debugger, code/security reviewers, ml_ai_advisor, ml_engineer, llm_engineer,
deep_learning_engineer, rl_engineer, cloud_engineer)
and `fast` (everyone else). Free tier: Groq/Gemini; keep `get_llm(tier)` the only place models
are created.

## Office view (game UI)
A Sims-style 2D top-down pixel office where you watch the team work: agents sit in team rooms,
walk to each other on handoffs and show what they're doing. Original art only (own sprites or a
free, permissively licensed office tileset); not a copy of any existing game.

- **Viewer, not engine.** The office only reads the event stream. The agent graph never imports
  or waits on it, runs headless without it, and the viewer makes no LLM calls.
- **Event format.** `runs/<id>/events.jsonl`, one JSON object per line: `{"t": secs, "type": ..., "agent": ...}`.
  | type | extra fields |
  |---|---|
  | `activate` | objective, reads |
  | `handoff` | from, to, objective |
  | `tool_call` | tool, arg (short) |
  | `artifact_written` | name |
  | `tests` | passed, failed |
  | `blocked` | questions |
  | `result` | status, summary, tokens |
  | `finish` | status, reason |
  The same events are emitted live via LangGraph's `get_stream_writer()` / `stream_mode="custom"`.
- **Rooms.** Leadership office (tech_lead), planning room with a whiteboard (the task board), build
  floor, QA lab, review meeting room, ops server room. Only agents the run wakes appear.
- **Event → animation.** handoff: the Tech Lead walks to the agent, speech bubble with the objective.
  activate/tool_call: typing, thought bubble with the tool and file. blocked: "?" over the head and
  a walk to the owner of the answer. tests failed: a bug sprite on the desk, the debugger visits.
  review: reviewers meet in the meeting room and return with ✅/🔴. Idle: coffee machine.
- **Game layer.** Per-agent stats (tasks done, fix loops, tokens, first-try streak); budget as an
  energy/clock bar; click a desk to read the agent's artifacts; you appear as the client, and plan
  approval happens in the meeting room.
- **Files.** `office/index.html`, `office/main.js` (Phaser 3, exact version from cdnjs),
  `office/assets/`, `office/sample-events.jsonl` (for building the UI without LLM calls);
  `devteam/events.py`; `server.py` (live mode only).
- **Modes.** Replay (open a run's events.jsonl, works offline) first; live (WebSocket) from step 15.

## Build order (commit after each; check before moving on)
11. **Registry + builder.** Load agents.yaml, validate, build 3 agents without tools.
    Check: each agent answers a hard-coded input in the required JSON format.
12. **Workspace tools.** Check: path escape rejected; bad Python edit rejected and file unchanged;
    viewer pages correctly; search returns file list.
13. **Artifacts + TeamState + SQLite checkpointer + event log** (`devteam/events.py` writes
    `runs/<id>/events.jsonl`, see Office view). Check: kill a run midway and resume it; the run's
    events.jsonl validates against the event types.
14. **Tech Lead supervisor (handoff → Command).** Planning team only.
    Check: request → prd, domain_notes, design, tasks.json; log every routing decision.
14b. **Office replay viewer** (`office/`, Phaser 3): rooms per team, avatars at desks, pathfinding
    between desks, speech/thought bubbles, playback controls (play/pause/speed/scrub).
    Check: replaying the step 14 run's events.jsonl shows every handoff as a walk + bubble.
15. **Human approval with interrupt()** after the task board (`--approve-plan`).
    Live office: `server.py` (FastAPI WebSocket) streams events while the graph runs, and the
    meeting room's "Approve plan" button resumes the interrupt().
    Check: run pauses, you approve or edit (CLI or office), it continues.
16. **One builder + per-task quality loop** (test_engineer, backend_dev, qa_runner, debugger).
    Check: a 3-task backend app goes green task by task.
17. **All builders + domain advisors.** Check: Tech Lead picks the right advisor/builder.
18. **Review stage** (code_reviewer, security_reviewer, review-round cap).
    Check: plant a SQL-injection bug; security_reviewer fails it; fix goes to the right owner.
19. **Parallel tasks with Send** (independent tasks, max 2 concurrent). Check: logs show overlap,
    no file conflicts.
19b. **Office game layer**: per-agent stats, budget energy/clock bar, clickable artifacts on desks,
    one animation per event type, parallel builders visibly working at once.
    Check: a full run replays with stats that match the run's activation log.
20. **Ops** (devops, docs_writer). Check: generated project installs, tests pass, smoke check runs.
21. **Stack profiles + Docker sandbox.** Load stacks.yaml; qa_runner/run_command use profile
    commands; non-Python stacks run in Docker. Check: the same small API built once as
    python-fastapi and once as node-express-ts both go green; react-vite-ts frontend builds.
22. **Services + presets.** devops starts the design's services with docker compose; architect can
    pick presets. Check: a PERN app's tests run against a real Postgres container.
23. **Specialist agents** (api_designer, ui_ux_designer, mobile_dev, deep_learning_engineer,
    rl_engineer, cloud_engineer, mlops_engineer, accessibility_reviewer). Add and test ONE at a time
    with its target app. Check: cloud_engineer never runs apply/deploy.
24. **Evaluation.** Run the 12 target apps (plus one mixed-stack app: FastAPI api + React web); record pass rate, activations, loops, tokens, time in
    RESULTS_v2.md. Tune prompts in agents.yaml based on failures, not guesses.

## Target apps (small, all external services mocked)
1. E-commerce: product catalogue + cart + checkout with mock payment, stock checks
2. SaaS: multi-tenant notes API with roles (admin/member) and plan-based note limits
3. Services: salon booking API with staff schedules and no double-booking
4. Data analysis: sales CSV → cleaned data, 5 metrics, 3 charts, summary JSON
5. IDP: classify sample invoices/receipts (text fixtures), extract fields, validate totals,
   low-confidence → review queue
6. ML: customer-churn model on synthetic data (baseline vs model, saved model, /predict API)
7. AI/LLM: RAG Q&A over a folder of markdown docs, answers with sources, "not found" when
   unsupported, FakeLLM in tests, small golden-set eval script

8. Mobile: Expo app (list + detail + form) against a FastAPI API (preset Mobile-Expo-API)
9. Deep learning: small CNN on a toy image dataset, beats baseline, saved checkpoint + report
10. Reinforcement learning: custom Gymnasium env + PPO agent that beats a random policy
11. Cloud: Terraform for an app (network, container service, Postgres, bucket), validated + tested
12. Full-stack preset: MERN or T3 app with Docker compose services and GitHub Actions CI

ML/AI apps need extra deps in the generated project: scikit-learn, pandas, joblib
(and the LLM provider package for the real adapter only).

## Conventions
- Prompts live in agents.yaml only; code never hard-codes role prompts
- Every agent activation is logged: agent, objective, artifacts read, result status, steps, tokens
- Don't add agents without a named failure they fix
- Don't build ahead of the current step
