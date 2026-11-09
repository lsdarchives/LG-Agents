# DevTeam v2: from a coding loop to an autonomous software team

This guide continues where the v1 guide ([LEARNING.md](LEARNING.md)) ends. v1 built one fixed
loop: four LLM agents and a test runner fixing a single Python file. v2 turns that into a **team**:
a Tech Lead supervisor and specialist agents that plan, build, test, review and ship small
multi-file projects, with every agent defined in a YAML file instead of in code.

You should have finished v1 (or at least read its Parts 1 and 2). This guide reuses its
concepts (state, nodes, conditional edges, retry and backoff, ground truth from real test runs)
without re-explaining them.

## Contents

- [What v2 adds](#what-v2-adds)
- [New concepts](#new-concepts)
- [Build it, step by step](#build-it-step-by-step)
- [FAQ](#faq)
- [Glossary](#glossary)

## What v2 adds

| v1 | v2 |
|---|---|
| 4 agents written as Python functions | 32 agents described in `agents.yaml`, built by one generic builder |
| Python routing functions pick the next node | A Tech Lead agent picks the next agent (supervisor pattern) |
| Agents return a text blob | Agents call guarded tools (file viewer, editor, test runner) |
| `solution.py` + `test_solution.py` | Multi-file projects in a sandboxed `workspace/` folder |
| One flat state, overwritten each step | Structured artifacts (PRD, design, task board, reports) and reducers |
| One model | Two model tiers: `strong` for judgement-heavy roles, `fast` for the rest |

## New concepts

### Config-driven agents (a registry)
In v1, adding an agent meant writing a function, a prompt constant and a graph edge. In v2 an
agent is **data**: a YAML entry with its name, team, model tier, which artifacts it reads, which
it writes, which tools it may use, a step budget and its prompt. One builder function turns any
entry into a working agent, so adding a role or tuning a prompt doesn't touch Python code.

### Validate at load time
A config file is only as good as its checks. Every entry is parsed into a Pydantic model with
`extra="forbid"`, so a misspelt field (`max_step`), an unknown team, an unknown tool name or a
missing prompt fails **when the file loads**, with the agent's name in the error. Without this,
the typo would surface twenty minutes into a run, when that agent is first woken.

### Model tiers
Agents don't name models. They say `model_tier: strong` or `fast`, and one config file maps each
tier to a real model per provider. The architect and reviewers get the strong model; routine
writers get the fast one. Swapping providers or models is still a one-line change.

### The result contract
Every LLM agent ends its reply with the same JSON block:
`{"status": "done" | "blocked" | "failed", "summary": "...", "questions": [], "handoff_notes": []}`.
That's what the supervisor reads to decide what happens next. `blocked` with `questions` is the
agent's way to ask instead of guessing (inventing missing requirements is a top cause of
multi-agent failure).

### Context isolation via `reads`
An agent sees **only** the artifacts listed in its `reads`. The debugger gets the failing test
report and the related files, not the PRD's marketing paragraph. That keeps prompts small (fewer
tokens, fewer rate-limit hits) and stops agents drifting into work that belongs to another role.

## Build it, step by step

Steps are numbered after v1's 1–10.

---

### Step 11: Agent registry and builder

**Goal:** load `agents.yaml`, validate every entry, and turn entries into callable agents.
No tools and no graph yet: each agent is one prompt-in, reply-out call.

**Concepts:** [config-driven agents](#config-driven-agents-a-registry),
[validate at load time](#validate-at-load-time), [model tiers](#model-tiers),
[the result contract](#the-result-contract), [context isolation](#context-isolation-via-reads).

**Check:** three agents (two tiers, two teams) answer hard-coded inputs and each reply contains a
valid result block.

#### What an entry looks like

```yaml
shared_rules: |
  You are one member of an autonomous software team. ...
  Every response ends with this result block, as the LAST thing in your reply, ALWAYS, even when
  your role asks for its own JSON (diagnosis, review, findings): give that first, then this block:
  {"status": "done" | "blocked" | "failed", "summary": "...", "questions": [], "handoff_notes": []}

agents:
  - name: debugger
    team: quality                 # leadership | domain | build | quality | ops
    kind: llm                     # llm, or tool (plain Python, no model)
    model_tier: strong            # mapped to a real model in config.py
    description: "Diagnoses failing tests or lint and says who should fix what."
    reads: [task, test_report, test_files, related_files, design]
    writes: [diagnosis]
    tools: [view_file, search_files, run_tests]
    max_steps: 8                  # tool-call budget for one activation
    prompt: |
      You are a senior debugging specialist. Find the ROOT cause of the failure, not the symptom.
      ...
```

`shared_rules` is put in front of every LLM agent's own prompt, so team-wide rules live in one place.

#### Config: tiers to models

Create `devteam/config.py`. It's the only file that names real models.

```python
"""v2 settings: provider, model tiers and retry policy. Models are named here and nowhere else."""

from pathlib import Path

# Switch provider here only: "groq" (primary) or "gemini" (fallback).
PROVIDER = "groq"

# agents.yaml says `model_tier: strong | fast`; this maps a tier to a real model per provider.
MODEL_TIERS = {
    "groq": {"strong": "openai/gpt-oss-120b", "fast": "openai/gpt-oss-20b"},
    "gemini": {"strong": "gemini-3.5-flash", "fast": "gemini-3.5-flash"},
}

RETRY_WAITS = [10, 20, 40]  # seconds to wait after each 429/503 before giving up
REQUEST_TIMEOUT = 120  # seconds per LLM request

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_FILE = REPO_ROOT / "agents.yaml"
```

#### LLM access with retry

Create `devteam/llm.py`. It's v1's retry and backoff logic (10s, 20s, 40s on 429/503; fail at once
on a daily quota) with one change: `get_llm(tier)` caches one model per tier.

```python
"""v2 model access: get_llm(tier) is the only place chat models are created."""

import time
from functools import cache
from typing import Literal

from dotenv import load_dotenv
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from devteam.config import MODEL_TIERS, PROVIDER, REQUEST_TIMEOUT, RETRY_WAITS

Tier = Literal["strong", "fast"]


def model_name(tier: Tier) -> str:
    return MODEL_TIERS[PROVIDER][tier]


@cache
def get_llm(tier: Tier):
    """Create one chat model per tier (cached). Library retries are off; invoke() retries."""
    load_dotenv()  # reads GROQ_API_KEY / GOOGLE_API_KEY from .env
    if PROVIDER == "groq":
        return ChatGroq(
            model=model_name(tier),
            temperature=0.2,
            reasoning_effort="low",
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )
    return ChatGoogleGenerativeAI(
        model=model_name(tier),
        temperature=0.2,
        thinking_level="low",
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


def _should_retry(exc: Exception) -> bool:
    """True for temporary 'slow down' errors: 429 quota and 503 model overloaded."""
    text = str(exc)
    return isinstance(exc, ModelRateLimitError) or any(
        code in text for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")
    )


def _is_daily_quota(exc: Exception) -> bool:
    """True if the 429 is a per-day cap: waiting seconds won't help."""
    return "PerDay" in str(exc) or "per day" in str(exc).lower()


def invoke(tier: Tier, system: str, user: str) -> str:
    """Send one system + user message to the tier's model, retrying 429/503. Returns reply text."""
    llm = get_llm(tier)
    messages = [SystemMessage(system), HumanMessage(user)]
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            return llm.invoke(messages).text
        except Exception as exc:
            if not _should_retry(exc):
                raise
            if _is_daily_quota(exc):
                raise RuntimeError(
                    f"Daily quota for {model_name(tier)} is used up; switch PROVIDER or "
                    "MODEL_TIERS in devteam/config.py, or wait for the reset."
                ) from exc
            if attempt == len(RETRY_WAITS):
                raise RuntimeError(
                    f"Still rate-limited/overloaded after {len(RETRY_WAITS)} retries. "
                    f"Last error: {exc}"
                ) from exc
            wait = RETRY_WAITS[attempt]
            print(f"  [retry] {type(exc).__name__}: waiting {wait}s ({attempt + 1}/{len(RETRY_WAITS)})")
            time.sleep(wait)
    raise AssertionError("unreachable")
```

#### The registry

Create `devteam/registry.py`. `AgentSpec` is one YAML entry; `Registry` is the whole file.

```python
"""Load and validate agents.yaml. A bad entry fails at load time, not mid-run."""

from collections import Counter
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from devteam.config import AGENTS_FILE

# Every tool name an agent may list. Implementations arrive in step 12+; listing them here
# means a typo like "view_files" is caught now instead of when the agent first runs.
KNOWN_TOOLS = frozenset(
    {
        "list_tree",
        "view_file",
        "search_files",
        "create_file",
        "edit_file",
        "run_tests",
        "run_command",
        "write_artifact",
        "read_artifact",
        "handoff",
    }
)


class AgentSpec(BaseModel):
    """One entry under `agents:` in agents.yaml."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    team: Literal["leadership", "domain", "build", "quality", "ops"]
    kind: Literal["llm", "tool"]
    description: str = Field(min_length=10)
    reads: list[str] = []
    writes: list[str] = []
    model_tier: Literal["strong", "fast"] | None = None
    tools: list[str] = []
    max_steps: int | None = Field(default=None, ge=1, le=50)
    prompt: str = ""

    @model_validator(mode="after")
    def _check_kind(self) -> "AgentSpec":
        if self.kind == "llm":
            missing = [f for f in ("model_tier", "max_steps") if getattr(self, f) is None]
            if not self.prompt.strip():
                missing.append("prompt")
            if missing:
                raise ValueError(f"llm agent {self.name!r} is missing {missing}")
        elif self.model_tier or self.prompt or self.tools:
            raise ValueError(f"tool agent {self.name!r} must not set model_tier, prompt or tools")
        unknown = sorted(set(self.tools) - KNOWN_TOOLS)
        if unknown:
            raise ValueError(f"agent {self.name!r} uses unknown tools {unknown}")
        return self


class Registry(BaseModel):
    """The whole agents.yaml file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shared_rules: str = Field(min_length=20)
    agents: list[AgentSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> "Registry":
        dupes = [n for n, c in Counter(a.name for a in self.agents).items() if c > 1]
        if dupes:
            raise ValueError(f"duplicate agent names: {dupes}")
        return self

    def get(self, name: str) -> AgentSpec:
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise KeyError(f"no agent named {name!r} in agents.yaml")


@cache
def load_registry(path: Path = AGENTS_FILE) -> Registry:
    """Parse and validate agents.yaml (cached per path)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Registry.model_validate(data)


if __name__ == "__main__":
    registry = load_registry()
    teams = Counter(a.team for a in registry.agents)
    kinds = Counter(a.kind for a in registry.agents)
    print(f"agents.yaml OK: {len(registry.agents)} agents")
    print("  by team:", dict(teams))
    print("  by kind:", dict(kinds))
```

The `_check_kind` validator encodes the rules the schema alone can't express: an `llm` agent needs a
tier, a step budget and a prompt, and a `tool` agent must not have them. `KNOWN_TOOLS` lists tool
names before the tools exist, so typos are caught now.

#### The builder

Create `devteam/builder.py`. `build_agent` joins the shared rules and the role prompt into the system
message, renders only the agent's `reads` into the user message, calls the tier's model, and parses
the result block.

```python
"""Turn an AgentSpec from agents.yaml into a callable agent.

Step 11: llm agents are a single prompt -> reply call (no tools yet). Each agent sees only the
artifacts named in its `reads`, and its reply must end with the shared JSON result block.
"""

import json
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ValidationError

from devteam.llm import invoke
from devteam.registry import AgentSpec, load_registry

MISSING = "(not provided)"


class AgentResult(BaseModel):
    """The JSON block every llm agent ends its reply with (see shared_rules in agents.yaml)."""

    status: Literal["done", "blocked", "failed"]
    summary: str
    questions: list[str] = []
    handoff_notes: list[str] = []


class AgentOutputError(ValueError):
    """The reply had no valid result block."""


def parse_result(text: str) -> AgentResult:
    """Return the LAST JSON object in `text` that has a "status" key, validated as AgentResult.

    Reviewers emit their own JSON (verdict/findings) before the result block, so we scan every
    '{' and keep the last object that looks like a result.
    """
    decoder = json.JSONDecoder()
    found = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "status" in obj:
            found = obj
    if found is None:
        raise AgentOutputError("reply has no JSON block with a 'status' key")
    try:
        return AgentResult.model_validate(found)
    except ValidationError as exc:
        raise AgentOutputError(f"result block is invalid: {exc}") from exc


def render_inputs(spec: AgentSpec, inputs: dict[str, str]) -> str:
    """Build the user message from the agent's `reads` only; anything else in `inputs` is dropped."""
    sections = [f"## {name}\n{inputs.get(name) or MISSING}" for name in spec.reads]
    return "\n\n".join(sections)


def build_agent(spec: AgentSpec, shared_rules: str) -> Callable[[dict[str, str]], dict]:
    """Return agent(inputs) -> {"result": AgentResult, "raw": reply text}."""
    if spec.kind == "tool":
        raise NotImplementedError(f"{spec.name!r} is a tool node; tool nodes are built in later steps")
    system = f"{shared_rules.strip()}\n\n{spec.prompt.strip()}"

    def agent(inputs: dict[str, str]) -> dict:
        print(f"[{spec.name}] tier={spec.model_tier} reads={spec.reads}")
        raw = invoke(spec.model_tier, system, render_inputs(spec, inputs))
        result = parse_result(raw)
        print(f"[{spec.name}] status={result.status} summary={result.summary[:100]!r}")
        return {"result": result, "raw": raw}

    agent.__name__ = spec.name
    return agent


def get_agent(name: str) -> Callable[[dict[str, str]], dict]:
    """Convenience: look up `name` in agents.yaml and build it."""
    registry = load_registry()
    return build_agent(registry.get(name), registry.shared_rules)


# Step 11 check: three agents (two tiers, two teams) answer hard-coded inputs in the JSON format.
CHECK_INPUTS = {
    "product_manager": {
        "request": "A tiny API to shorten URLs: POST a long URL, get a short code; GET the code redirects.",
    },
    "architect": {
        "prd": (
            "Goal: shorten URLs. Story (MUST): As a user I POST a URL and get a 6-char code.\n"
            "AC: Given a valid URL, When I POST /links, Then I get 201 and a code.\n"
            "Story (MUST): GET /{code} redirects (307) to the URL; unknown code -> 404."
        ),
        "stack_catalog": "python-fastapi: use_when small JSON APIs; pytest + httpx TestClient; SQLite ok.",
    },
    "debugger": {
        "task": "T2: GET /{code} redirects to the stored URL; unknown code returns 404.",
        "test_report": (
            "FAILED tests/test_links.py::test_unknown_code_404 - assert 500 == 404\n"
            "app/services.py:14 in resolve: return LINKS[code]  KeyError: 'zzzzzz'"
        ),
        "test_files": (
            "def test_unknown_code_404(client):\n"
            "    assert client.get('/zzzzzz', follow_redirects=False).status_code == 404"
        ),
        "related_files": (
            "# app/services.py\nLINKS = {}\n\ndef resolve(code):\n    return LINKS[code]\n\n"
            "# app/main.py\n@app.get('/{code}')\ndef go(code):\n"
            "    return RedirectResponse(resolve(code), 307)"
        ),
    },
}

if __name__ == "__main__":
    failures = 0
    for agent_name, agent_inputs in CHECK_INPUTS.items():
        try:
            out = get_agent(agent_name)(agent_inputs)
            print(json.dumps(out["result"].model_dump(), indent=2)[:600], "\n")
        except AgentOutputError as exc:
            failures += 1
            print(f"[{agent_name}] FAILED: {exc}\n")
    print("step 11 check:", "PASS" if failures == 0 else f"{failures} agent(s) failed")
    raise SystemExit(1 if failures else 0)
```

**Design notes**
- **The last `status` JSON wins.** Reviewers and the debugger print their own JSON (findings,
  diagnosis) before the result block. `parse_result` tries `json.JSONDecoder.raw_decode` at every
  `{` and keeps the last object with a `status` key, so both blocks can live in one reply.
- **Missing inputs are explicit.** A read the caller didn't supply is rendered as `(not provided)`,
  which gives the agent a clear reason to return `blocked` instead of inventing the content.
- **Extra inputs are dropped.** Passing more artifacts than the agent `reads` has no effect; isolation
  is enforced by the builder, not by caller discipline.
- **Tool nodes are deferred.** `build_agent` refuses `kind: tool` for now; the test runner node
  arrives with the quality loop.
- **`llm.py` duplicates v1's retry code on purpose**, so the finished v1 code stays exactly as it
  was released. Once v1's `agents.py` is retired, `llm.py` is the only copy.

**Check**

Validate the registry (no API calls):

```powershell
.venv\Scripts\python -m devteam.registry
```

Expected: `agents.yaml OK: 32 agents`, with counts by team and kind.

Run three real agents (product_manager on the fast tier; architect and debugger on the strong tier):

```powershell
.venv\Scripts\python -X utf8 -m devteam.builder
```

Expected: each agent prints `status=done` with a summary, then `step 11 check: PASS`.

**Try it:** break `agents.yaml` in each of these ways and rerun the registry check. Each should
fail with the agent's name in the message:
- rename `max_steps` to `max_step` on any agent (extra field)
- add `view_files` to an agent's tools (unknown tool)
- change a `team` to `sales`
- give two agents the same `name`

**Check yourself**
1. Why does the builder render `(not provided)` for a missing read instead of leaving it out?
2. The debugger's prompt asks for a diagnosis JSON and the shared rules ask for a result block.
   Why scan for the *last* object with a `status` key rather than the first JSON in the reply?
3. What would go wrong if `extra="forbid"` were removed from `AgentSpec`?

**Gotchas**
- **Two output formats in one prompt conflict.** On the first live run, the debugger answered with
  only its diagnosis JSON and skipped the result block: its role prompt said "Output diagnosis
  JSON" and the model treated that as the whole answer. The fix was in `shared_rules`, not in the
  parser: state that the result block comes **last, always, even after role JSON**. The same applies
  to every reviewer. This is the kind of failure the step's check exists to catch.
- **Agents don't know each other's names yet.** The debugger may set `fix_owner: "dev"`, which isn't
  a real agent. The roster is given to the Tech Lead in the supervisor step.
- **`yaml.safe_load`, never `yaml.load`.** Plain `load` can construct arbitrary Python objects from
  a YAML file.
- On Windows, run the live check with `-X utf8` so model output with non-ASCII characters prints.

---

### Step 12: Workspace tools
_Written when the step is built._

### Step 13: Artifacts, TeamState and a SQLite checkpointer
_Written when the step is built._

### Step 14: Tech Lead supervisor (handoff → Command)
_Written when the step is built._

### Step 15: Human approval with interrupt()
_Written when the step is built._

### Step 16: One builder and the per-task quality loop
_Written when the step is built._

### Step 17: All builders and domain advisors
_Written when the step is built._

### Step 18: Review stage
_Written when the step is built._

### Step 19: Parallel tasks with Send
_Written when the step is built._

### Step 20: Ops (devops, docs_writer)
_Written when the step is built._

### Step 21: Stack profiles and Docker sandbox
_Written when the step is built._

### Step 22: Services and presets
_Written when the step is built._

### Step 23: Specialist agents
_Written when the step is built._

### Step 24: Evaluation
_Written when the step is built._

## FAQ

**Why YAML instead of Python classes for agents?**
Roles and prompts change far more often than the machinery that runs them. Keeping them as data
means tuning a prompt is a one-line diff with no risk to the code, and the whole team is readable
in one file.

**Why validate with Pydantic instead of just reading the dict?**
A dict lookup fails late (`KeyError` mid-run) and silently accepts typos as extra keys. A model with
`extra="forbid"` fails early, names the field and the agent, and documents the schema in code.

**Why two model tiers instead of one model for everything?**
Free-tier quotas are per model. Routing routine work to a smaller, faster model saves the strong
model's quota for the roles where judgement matters (architecture, debugging, review).

## Glossary

- **Registry:** the validated list of agent specs loaded from `agents.yaml`.
- **AgentSpec:** one agent's definition: name, team, kind, tier, reads, writes, tools, budget, prompt.
- **Builder:** the function that turns an AgentSpec into a callable agent.
- **Model tier:** an abstract model size (`strong` or `fast`) mapped to a real model in config.
- **Result block:** the JSON every agent ends with: status, summary, questions, handoff_notes.
- **Blocked:** an agent's status when required information is missing; it returns questions instead of guessing.
- **Reads / writes:** the artifacts an agent receives as input and is expected to produce.
- **Shared rules:** instructions prefixed to every LLM agent's prompt.
