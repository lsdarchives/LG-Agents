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

### Agent-computer interface (ACI)
An agent is only as good as the tools it can drive. Research on coding agents (SWE-agent) found that
small, guarded tools with compact output (a paged viewer, an edit command that rejects broken
syntax, a search that returns file names) beat giving the model a raw shell. Tools are an interface
designed for the model, the way a UI is designed for people.

### Tool-calling loop
The model is given tool schemas. Each turn it either asks for tool calls or replies with its final
answer. The program runs the requested tools, appends their results as tool messages, and asks again.
This is the ReAct pattern (reason, act, observe), capped at a fixed number of rounds.

### Path guard
Every path from an agent is resolved to an absolute path and must stay under the run's workspace
folder. Resolving first matters: `a/../../b.txt` looks relative but points outside.

### Reducers for team state
In v1 each field had one writer, so overwriting was fine. In a team, several agents touch the same
fields (the task board, the artifact index), sometimes in the same step. A reducer is the merge rule
LangGraph applies when a node returns a value for that field: merge by id, union of keys, or a sum.

### Checkpointer
A checkpointer saves the full graph state after every node, keyed by a thread id. Give the run id as
the thread id and any run can be inspected (`get_state`) or resumed (`invoke(None, config)`) after a
crash, a Ctrl+C, or a pause for human approval.

### Event log
A run's public record: one validated JSON event per line (who was activated, which tools they called,
what they wrote, how it ended). It is written for observers (the office view, dashboards, you)
and is separate from the checkpoint, which belongs to the engine.

### Output contract
An agent's word that it is "done" is not evidence. The builder checks the outputs an agent is
supposed to leave behind (its `writes`) and refuses "done" when they don't exist.

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
(Originally this was `python -m devteam.builder`; from step 12 the live checks live in `checks/`.)

```powershell
.venv\Scripts\python -X utf8 -m checks.step11
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

**Goal:** give agents hands. Each run gets a project folder, and agents act on it through a small set
of guarded tools: browse, read, search, create, edit, and save team documents. The builder turns
every LLM agent into a tool-calling loop.

**Concepts:** [agent-computer interface](#agent-computer-interface-aci),
[tool-calling loop](#tool-calling-loop), [path guard](#path-guard).

**Check:** a path that escapes the workspace is rejected; a bad Python edit is rejected and the file
is unchanged; the viewer pages correctly; search returns a file list. Live: product_manager writes
`prd.md` and architect writes `design.md`, both through `write_artifact`.

#### The workspace

Create `devteam/toolbox/workspace.py`. A `Workspace` is one run's folder. Every path an agent sends
goes through `resolve()`, which resolves `..` and absolute paths and rejects anything outside the
root. Team documents live in `.devteam/` and can only be reached with the artifact tools, so a
builder can't overwrite the PRD by accident.

```python
"""Workspace tools: the file operations agents use, all locked inside one run's folder.

Design (agent-computer interface): small outputs, line numbers, clear errors returned as text
(never raised at the agent), and edits that can't leave a Python file unparsable.
"""

import ast
import re
from pathlib import Path

from langchain_core.tools import ToolException

VIEW_LINES = 100  # lines per view_file page
MAX_OUTPUT_CHARS = 6000  # cap on any single tool reply
ARTIFACT_DIR = ".devteam"  # team documents (prd, design, ...) live here, not in the project tree
SKIP_DIRS = {ARTIFACT_DIR, ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
ARTIFACT_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
NO_OUTPUT = "Command ran successfully with no output."


class ToolError(ToolException):
    """A tool refused the request. LangChain turns it into a reply the agent sees, not a crash."""


def _trim(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text or NO_OUTPUT
    return text[:MAX_OUTPUT_CHARS] + f"\n... [trimmed {len(text) - MAX_OUTPUT_CHARS} chars]"


class Workspace:
    """One run's project folder. Every path an agent gives is resolved and checked against it."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ARTIFACT_DIR).mkdir(exist_ok=True)

    # ── path guard ────────────────────────────────────────────────────────────
    def resolve(self, rel: str) -> Path:
        """Return the absolute path for `rel`, or raise ToolError if it escapes the workspace."""
        if not rel or not rel.strip():
            raise ToolError("path is empty")
        path = (self.root / rel.strip()).resolve()
        if not path.is_relative_to(self.root):
            raise ToolError(f"path {rel!r} is outside the workspace; use paths relative to the project root")
        if ARTIFACT_DIR in path.relative_to(self.root).parts:
            raise ToolError(f"{ARTIFACT_DIR}/ holds team artifacts; use read_artifact / write_artifact")
        return path

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # ── read tools ────────────────────────────────────────────────────────────
    def list_tree(self, depth: int = 3) -> str:
        """Compact file tree, directories first, skipping caches, venvs and artifacts."""
        depth = max(1, min(int(depth), 6))
        lines: list[str] = []

        def walk(folder: Path, level: int) -> None:
            entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            for entry in entries:
                if entry.is_dir() and entry.name in SKIP_DIRS:
                    continue
                lines.append("  " * level + entry.name + ("/" if entry.is_dir() else ""))
                if entry.is_dir() and level + 1 < depth:
                    walk(entry, level + 1)

        walk(self.root, 0)
        return _trim("\n".join(lines)) if lines else "(workspace is empty)"

    def view_file(self, path: str, start: int = 1) -> str:
        """Show VIEW_LINES numbered lines from `start`, with the path and total line count."""
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"{path!r} is not a file; call list_tree to see what exists")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        start = max(1, int(start))
        if total and start > total:
            raise ToolError(f"start={start} is past the end; {path} has {total} lines")
        end = min(start + VIEW_LINES - 1, total)
        body = "\n".join(f"{n:>4} | {lines[n - 1]}" for n in range(start, end + 1))
        more = f"\n... {total - end} more lines; call view_file({path!r}, start={end + 1})" if end < total else ""
        return _trim(f"[{self.rel(target)}: lines {start}-{end} of {total}]\n{body}{more}")

    def search_files(self, pattern: str) -> str:
        """List files containing `pattern` (plain text, case-insensitive) with match counts."""
        if not pattern:
            raise ToolError("pattern is empty")
        needle = pattern.lower()
        hits: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(self.root).parts):
                continue
            try:
                count = path.read_text(encoding="utf-8").lower().count(needle)
            except (UnicodeDecodeError, OSError):
                continue
            if count:
                hits.append(f"{self.rel(path)} ({count} matches)")
        if not hits:
            return f"No files contain {pattern!r}."
        return _trim(f"{len(hits)} files contain {pattern!r}:\n" + "\n".join(hits))

    # ── write tools ───────────────────────────────────────────────────────────
    @staticmethod
    def _check_python(path: Path, text: str) -> None:
        if path.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as exc:
                raise ToolError(
                    f"rejected: the result would not parse (line {exc.lineno}: {exc.msg}); file unchanged"
                ) from exc

    def create_file(self, path: str, content: str) -> str:
        """Create a new file. Refuses to overwrite: existing files are changed with edit_file."""
        target = self.resolve(path)
        if target.exists():
            raise ToolError(f"{path!r} already exists; view it and use edit_file to change it")
        self._check_python(target, content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return f"Created {self.rel(target)} ({len(content.splitlines())} lines)."

    def edit_file(self, path: str, start_line: int, end_line: int, new_text: str) -> str:
        """Replace lines start_line..end_line (1-based, inclusive) with new_text.

        To insert without deleting, use end_line = start_line - 1. Python files are syntax-checked
        and the edit is rejected (file unchanged) if the result doesn't parse.
        """
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"{path!r} does not exist; use create_file")
        lines = target.read_text(encoding="utf-8").splitlines()
        start_line, end_line = int(start_line), int(end_line)
        if start_line < 1 or start_line > len(lines) + 1 or end_line < start_line - 1 or end_line > len(lines):
            raise ToolError(f"line range {start_line}-{end_line} is invalid; {path} has {len(lines)} lines")
        replacement = new_text.splitlines()
        updated = lines[: start_line - 1] + replacement + lines[end_line:]
        text = "\n".join(updated) + "\n"
        self._check_python(target, text)
        target.write_text(text, encoding="utf-8", newline="\n")
        # Show the edited region with a little context so the agent can verify the change.
        lo = max(1, start_line - 2)
        hi = min(len(updated), start_line + len(replacement) + 1)
        region = "\n".join(f"{n:>4} | {updated[n - 1]}" for n in range(lo, hi + 1))
        return _trim(f"Edited {self.rel(target)}; lines {lo}-{hi} now:\n{region}")

    # ── artifacts ─────────────────────────────────────────────────────────────
    def _artifact_path(self, name: str) -> Path:
        name = name.strip().lower()
        if not ARTIFACT_NAME.match(name):
            raise ToolError(f"artifact name {name!r} must be lowercase letters, digits, '_', '-' or '.'")
        if "." not in name:
            name += ".md"
        return self.root / ARTIFACT_DIR / name

    def write_artifact(self, name: str, content: str) -> str:
        """Save a team document (e.g. 'prd', 'design', 'tasks.json'). Overwrites the previous version."""
        if not content.strip():
            raise ToolError("artifact content is empty")
        path = self._artifact_path(name)
        path.write_text(content, encoding="utf-8", newline="\n")
        return f"Saved artifact {path.name} ({len(content.splitlines())} lines)."

    def read_artifact(self, name: str) -> str:
        """Return a team document's full text."""
        path = self._artifact_path(name)
        if not path.exists():
            have = sorted(p.name for p in (self.root / ARTIFACT_DIR).iterdir())
            raise ToolError(f"no artifact {path.name!r}; existing: {have or 'none'}")
        return _trim(path.read_text(encoding="utf-8"))

    def artifacts(self) -> dict[str, str]:
        """name -> relative path of every saved artifact (for the artifact index)."""
        folder = self.root / ARTIFACT_DIR
        return {p.stem: self.rel(p) for p in sorted(folder.iterdir()) if p.is_file()}
```

The folder is called `toolbox` rather than `tools` because v1 already has `devteam/tools.py`, and a
package with the same name would shadow it.

#### Turning methods into LangChain tools

Create `devteam/toolbox/__init__.py`. Each tool is a small function with type hints and a docstring;
`StructuredTool.from_function` turns those into the JSON schema the model sees. `ToolError` subclasses
LangChain's `ToolException`, so `handle_tool_error` sends a refusal back to the agent as
`ERROR: ...` text instead of crashing the run. `handle_validation_error` does the same for bad arguments.

```python
"""v2 agent tools. Named `toolbox` so it doesn't shadow v1's devteam/tools.py.

make_tools(workspace, names) turns tool names from agents.yaml into LangChain tools bound to one
workspace. A refused request (ToolError) or bad arguments come back to the agent as an
"ERROR: ..." reply it can read and react to, instead of crashing the run.
"""

from langchain_core.tools import BaseTool, StructuredTool

from devteam.toolbox.workspace import ToolError, Workspace


def _error_text(exc: Exception) -> str:
    return f"ERROR: {exc}"


def _bad_args_text(exc: Exception) -> str:
    return f"ERROR: invalid arguments: {exc}"


def workspace_tools(ws: Workspace) -> dict[str, BaseTool]:
    """All workspace tools, bound to `ws`, keyed by the names agents.yaml uses."""

    def list_tree(depth: int = 3) -> str:
        """Show the project's file tree (directories first), `depth` levels deep."""
        return ws.list_tree(depth)

    def view_file(path: str, start: int = 1) -> str:
        """Show 100 numbered lines of a project file, starting at line `start`."""
        return ws.view_file(path, start)

    def search_files(pattern: str) -> str:
        """List project files containing `pattern` (plain text, case-insensitive) with match counts."""
        return ws.search_files(pattern)

    def create_file(path: str, content: str) -> str:
        """Create a NEW project file with `content`. Fails if it exists (then use edit_file)."""
        return ws.create_file(path, content)

    def edit_file(path: str, start_line: int, end_line: int, new_text: str) -> str:
        """Replace lines start_line..end_line (1-based, inclusive) of a file with new_text.
        To insert without deleting use end_line = start_line - 1. Python edits that break syntax are rejected."""
        return ws.edit_file(path, start_line, end_line, new_text)

    def write_artifact(name: str, content: str) -> str:
        """Save a team document by name (e.g. 'prd', 'design', 'tasks.json'); overwrites the old version."""
        return ws.write_artifact(name, content)

    def read_artifact(name: str) -> str:
        """Read a team document by name (e.g. 'prd', 'design')."""
        return ws.read_artifact(name)

    fns = [list_tree, view_file, search_files, create_file, edit_file, write_artifact, read_artifact]
    return {
        fn.__name__: StructuredTool.from_function(
            func=fn,
            name=fn.__name__,
            description=fn.__doc__,
            handle_tool_error=_error_text,
            handle_validation_error=_bad_args_text,
        )
        for fn in fns
    }


def make_tools(ws: Workspace, names: list[str]) -> tuple[list[BaseTool], list[str]]:
    """Return (tools for `names` that exist, names not implemented yet)."""
    available = workspace_tools(ws)
    tools = [available[n] for n in names if n in available]
    missing = [n for n in names if n not in available]
    return tools, missing


__all__ = ["ToolError", "Workspace", "make_tools", "workspace_tools"]
```

#### The tool-calling loop

`devteam/builder.py` now binds an agent's tools to its model and loops: the model either calls tools
(the results are appended as `ToolMessage`s) or replies with its final answer. The loop is capped at
`max_steps` rounds. When the budget runs out, every pending call gets an "ERROR: tool budget used up"
answer and the model must reply without tools. If the final reply has no result block, the builder
asks once for just the block.

```python
"""Turn an AgentSpec from agents.yaml into a callable agent.

An llm agent is a small tool-calling loop: the model either calls tools (bound to one workspace)
or gives its final answer. The loop is capped at the agent's `max_steps` tool rounds, and the
final reply must end with the shared JSON result block.
"""

import json
from collections.abc import Callable
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from devteam.llm import get_llm, invoke_messages
from devteam.registry import AgentSpec, load_registry
from devteam.toolbox import Workspace, make_tools

MISSING = "(not provided)"
BUDGET_NOTE = (
    "Your tool budget for this activation is used up. Do not call any more tools. "
    "Reply now with your final answer, ending with the result block."
)
REPAIR_NOTE = (
    "Your reply did not end with a valid result block. Reply again with only the result block: "
    '{"status": "done" | "blocked" | "failed", "summary": "...", "questions": [], "handoff_notes": []}'
)


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


def system_prompt(spec: AgentSpec, shared_rules: str) -> str:
    """Shared rules + role prompt + what this activation must produce and with which budget."""
    work_note = (
        f"Your outputs: {', '.join(spec.writes) or 'none'}. Produce them with your tools "
        f"({', '.join(spec.tools)}); artifacts go through write_artifact, project files through "
        f"create_file/edit_file. You have at most {spec.max_steps} rounds of tool calls."
    )
    return f"{shared_rules.strip()}\n\n{spec.prompt.strip()}\n\n{work_note}"


def _short(args: dict, limit: int = 80) -> str:
    """One-line summary of tool arguments for logs (long content is elided)."""
    parts = []
    for key, value in args.items():
        text = str(value).replace("\n", " ")
        parts.append(f"{key}={text[:40] + '...' if len(text) > 40 else text!r}")
    line = ", ".join(parts)
    return line if len(line) <= limit else line[: limit - 3] + "..."


def build_agent(spec: AgentSpec, shared_rules: str) -> Callable[[dict[str, str], Workspace], dict]:
    """Return agent(inputs, workspace) -> {"result", "raw", "steps", "tool_calls"}."""
    if spec.kind == "tool":
        raise NotImplementedError(f"{spec.name!r} is a tool node; tool nodes are built in later steps")
    system = system_prompt(spec, shared_rules)

    def agent(inputs: dict[str, str], workspace: Workspace) -> dict:
        tools, missing = make_tools(workspace, spec.tools)
        by_name = {t.name: t for t in tools}
        plain = get_llm(spec.model_tier)
        model = plain.bind_tools(tools) if tools else plain
        print(f"[{spec.name}] tier={spec.model_tier} reads={spec.reads} tools={list(by_name)}"
              + (f" (not built yet: {missing})" if missing else ""))

        messages = [SystemMessage(system), HumanMessage(render_inputs(spec, inputs))]
        tool_calls: list[dict] = []
        steps = 0
        reply = invoke_messages(model, messages, spec.model_tier, set(by_name))
        messages.append(reply)
        while reply.tool_calls:
            if steps >= spec.max_steps:
                for call in reply.tool_calls:  # every tool call needs an answer before the next turn
                    messages.append(ToolMessage("ERROR: tool budget used up", tool_call_id=call["id"]))
                messages.append(HumanMessage(BUDGET_NOTE))
                reply = invoke_messages(plain, messages, spec.model_tier)
                messages.append(reply)
                break
            steps += 1
            for call in reply.tool_calls:
                tool = by_name.get(call["name"])
                if tool is None:
                    out = ToolMessage(
                        f"ERROR: unknown tool {call['name']!r}; you have {list(by_name)}",
                        tool_call_id=call["id"],
                    )
                else:
                    out = tool.invoke(call)  # returns a ToolMessage; errors come back as "ERROR: ..."
                tool_calls.append({"tool": call["name"], "args": call["args"]})
                first = str(out.content).splitlines()[0] if out.content else ""
                print(f"  [{spec.name}] {call['name']}({_short(call['args'])}) -> {first[:90]}")
                messages.append(out)
            reply = invoke_messages(model, messages, spec.model_tier, set(by_name))
            messages.append(reply)

        raw = reply.text
        try:
            result = parse_result(raw)
        except AgentOutputError:
            messages.append(HumanMessage(REPAIR_NOTE))  # one repair attempt, then give up
            raw = invoke_messages(plain, messages, spec.model_tier).text
            result = parse_result(raw)
        print(f"[{spec.name}] status={result.status} steps={steps} summary={result.summary[:100]!r}")
        return {"result": result, "raw": raw, "steps": steps, "tool_calls": tool_calls}

    agent.__name__ = spec.name
    return agent


def get_agent(name: str) -> Callable[[dict[str, str], Workspace], dict]:
    """Convenience: look up `name` in agents.yaml and build it."""
    registry = load_registry()
    return build_agent(registry.get(name), registry.shared_rules)
```

#### Recovering a phantom tool call

`devteam/llm.py` gains `invoke_messages`, the retrying call used by the loop, and one provider fix:

```python
def _recover_phantom_tool_call(exc: Exception, tool_names: set[str]) -> AIMessage | None:
    """Groq + GPT-OSS sometimes sends its final JSON answer as a call to a tool that doesn't exist
    (e.g. "json"); Groq rejects it with 400 tool_use_failed. The rejected call's arguments ARE the
    intended answer, so return them as a plain text reply. Calls to real tools are not recovered."""
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(error, dict) or error.get("code") != "tool_use_failed":
        return None
    raw = error.get("failed_generation") or ""
    try:
        generation = json.loads(raw)
        name = generation.get("name") if isinstance(generation, dict) else None
        content = json.dumps(generation.get("arguments", generation)) if name else None
    except json.JSONDecodeError:
        # e.g. two JSON objects inside "arguments": keep the raw text; parse_result finds the result.
        match = re.match(r'\s*\{\s*"name"\s*:\s*"([^"]+)"', raw)
        name, content = (match.group(1), raw) if match else (None, None)
    if not name or name in tool_names:
        return None
    print(f"  [llm] recovered final answer sent as phantom tool {name!r}")
    return AIMessage(content=content)
```

**Design notes**
- **Errors are replies, not exceptions.** An agent that gets `ERROR: path 'x' is outside the workspace`
  can correct itself; an exception would end the activation and waste every call made so far.
- **`create_file` refuses to overwrite.** Changing a file goes through `edit_file`, which shows the
  edited region back. That makes accidental full-file rewrites (the v1 lint failure mode) less likely.
- **Syntax-checked edits.** For `.py` files, both `create_file` and `edit_file` run `ast.parse` on the
  result and refuse it if it doesn't parse, so an agent can never leave a half-edited file behind.
- **Small, bounded output.** `view_file` pages 100 lines; `search_files` returns only file names
  and counts; every reply is capped at 6000 characters. Agents read what they need instead of
  having whole files pasted in.
- **Unbuilt tools are skipped.** `make_tools` returns the names it can't build yet (`run_tests`,
  `handoff`, ...), and the agent runs with the rest.
- **Our own loop instead of a prebuilt ReAct agent.** It's 40 lines, it uses our retry wrapper on
  every call, enforces `max_steps` exactly, and gives one place to emit `tool_call` events later
  (for the office view).

**Check**

Offline (no API calls):

```powershell
.venv\Scripts\python -m pytest checks -q
```

Expected: all tests pass, including `checks/test_workspace.py`.

Live (needs `.env`):

```powershell
.venv\Scripts\python -X utf8 -m checks.phase01
```

Expected: product_manager calls `write_artifact` for `prd.md`, architect writes `design.md`, both
artifacts are printed, then `phase 1 check: PASS`. The files stay in `workspace/check-phase01/.devteam/`
(gitignored) for you to read.

**Try it**
- In a Python shell, create `Workspace("workspace/try")` and call `create_file("../escape.txt", "x")`,
  `view_file` on a 300-line file, and an `edit_file` that deletes a closing bracket in a `.py` file.
- Call a tool through LangChain (`make_tools(ws, ["view_file"])[0][0].invoke({"path": "nope.py"})`)
  and compare the reply with the exception the method raises directly.

**Check yourself**
1. Why does `resolve()` call `.resolve()` *before* checking `is_relative_to`?
2. What would an agent see, and do, if a tool raised a plain `ValueError` instead of `ToolError`?
3. Why must every pending tool call get a `ToolMessage` before the "budget used up" note?

**Gotchas**
- **Phantom tool calls.** GPT-OSS on Groq sometimes sends its final JSON answer as a call to a tool
  that doesn't exist (`json`, `JSON`, `diagnosis`), and Groq rejects it with 400 `tool_use_failed`.
  The rejected generation *is* the answer, so `_recover_phantom_tool_call` turns it into a text
  reply. Calls to real tools with bad arguments are still errors.
- **Raw strings for Windows paths.** In Python source (and in docstrings that show
  `.venv\Scripts\...`), use `r"..."`, or `\S`/`\x` become invalid escape sequences.
- **Rate limits add up.** A tool-using agent makes one model call per tool round, so a 3-round
  activation costs 4 calls instead of 1. The retry wrapper absorbs the free tier's 429s, but keep
  `max_steps` small.

---

### Step 13: Artifacts, TeamState and a SQLite checkpointer

**Goal:** give the team a shared memory that survives crashes, and a record of everything it does.
Agents get their `reads` from state plus the workspace's artifacts, every activation is logged as
events (which the office view will replay), and a run can be killed and resumed by its id.

**Concepts:** [reducers for team state](#reducers-for-team-state), [checkpointer](#checkpointer),
[event log](#event-log), [output contract](#output-contract).

**Check:** kill a run after its 2nd agent, resume it, and it finishes without rerunning finished
agents; every line of its `events.jsonl` validates.

#### Limits and paths

`devteam/config.py` gains where runs live and the run-wide caps later steps enforce:

```python
# Where each run lives: project files in workspace/<run_id>/, logs + events in runs/<run_id>/.
WORKSPACE_ROOT = REPO_ROOT / "workspace"
RUNS_ROOT = REPO_ROOT / "runs"
CHECKPOINT_DB = RUNS_ROOT / "checkpoints.sqlite"  # SQLite checkpointer: resume a run by its id

# Hard limits (autonomous routing can loop forever without them).
RECURSION_LIMIT = 200  # LangGraph super-steps per invoke
MAX_ACTIVATIONS = 60  # agent activations per run
MAX_FIX_LOOPS = 4  # debugger -> fix -> qa rounds per task
MAX_REVIEW_ROUNDS = 2
MAX_BLOCKED_CYCLES = 3  # blocked -> answer cycles per run
MAX_PARALLEL_BUILDERS = 2  # free-tier rate limits
```

#### TeamState with reducers

Create `devteam/team_state.py` (v1 owns `state.py`). Fields that several nodes write get a reducer.

```python
"""v2 shared state. Reducers let several nodes contribute to the same field safely.

(Named team_state.py because v1 owns devteam/state.py.)
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages


class Task(TypedDict, total=False):
    id: str
    title: str
    owner_role: str
    component: str
    stack: str
    files: list[str]
    depends_on: list[str]
    acceptance: list[str]
    status: Literal["todo", "in_progress", "done", "failed"]


def merge_dicts(old: dict[str, str] | None, new: dict[str, str] | None) -> dict[str, str]:
    """Reducer: later keys win; nothing is ever dropped."""
    return {**(old or {}), **(new or {})}


def merge_tasks(old: list[Task] | None, new: list[Task] | None) -> list[Task]:
    """Reducer: merge tasks by id. An update for an existing id changes only the fields it sends;
    new ids are appended in order. Two parallel builders can each update their own task."""
    merged: dict[str, Task] = {t["id"]: dict(t) for t in (old or [])}
    for task in new or []:
        merged[task["id"]] = {**merged.get(task["id"], {}), **task}
    return list(merged.values())


class TeamState(TypedDict):
    request: str
    run_id: str
    messages: Annotated[list, add_messages]  # audit log of agent results (not shared context)
    artifacts: Annotated[dict[str, str], merge_dicts]  # artifact name -> path in the workspace
    task_board: Annotated[list[Task], merge_tasks]
    current_task: str | None
    last_result: dict  # last agent's result block + its name
    activations: Annotated[int, operator.add]  # each node returns 1; the reducer sums
    review_rounds: int
    status: Literal["running", "done", "partial", "failed"]
    reason: str


def initial_team_state(request: str, run_id: str) -> TeamState:
    return {
        "request": request,
        "run_id": run_id,
        "messages": [],
        "artifacts": {},
        "task_board": [],
        "current_task": None,
        "last_result": {},
        "activations": 0,
        "review_rounds": 0,
        "status": "running",
        "reason": "",
    }
```

- `merge_tasks` merges by id, so a builder can send `{"id": "T2", "status": "done"}` without
  resending the whole board, and two parallel builders can update different tasks in one step.
- `activations` uses `operator.add`: each node returns `1` and LangGraph sums them.
- `messages` (`add_messages`) is an audit log of results. It is never sent to agents as context.

#### The event log

Create `devteam/events.py`. Each event type is a Pydantic model, combined into a discriminated union
on `type`, so an unknown type or a bad field fails when it is written, not when the office
tries to draw it.

```python
"""Event log: what happened in a run, as one JSON object per line in runs/<id>/events.jsonl.

The office view replays this file (and receives the same events live through LangGraph's custom
stream). The engine only writes events; it never reads them back or depends on a viewer.
"""

import json
import time
from pathlib import Path
from typing import Annotated, Literal

from langgraph.config import get_stream_writer
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    t: float  # seconds since the run started (replay timeline)
    ts: float  # wall-clock epoch seconds


class Activate(_Event):
    type: Literal["activate"] = "activate"
    agent: str
    objective: str
    reads: list[str] = []


class Handoff(_Event):
    type: Literal["handoff"] = "handoff"
    from_: str = Field(alias="from")
    to: str
    objective: str


class ToolCall(_Event):
    type: Literal["tool_call"] = "tool_call"
    agent: str
    tool: str
    arg: str = ""  # short summary of the arguments


class ArtifactWritten(_Event):
    type: Literal["artifact_written"] = "artifact_written"
    agent: str
    name: str


class Tests(_Event):
    type: Literal["tests"] = "tests"
    agent: str
    passed: int
    failed: int


class Blocked(_Event):
    type: Literal["blocked"] = "blocked"
    agent: str
    questions: list[str]


class Result(_Event):
    type: Literal["result"] = "result"
    agent: str
    status: Literal["done", "blocked", "failed"]
    summary: str
    steps: int = 0
    tokens: int = 0


class Finish(_Event):
    type: Literal["finish"] = "finish"
    status: str
    reason: str = ""


Event = Annotated[
    Activate | Handoff | ToolCall | ArtifactWritten | Tests | Blocked | Result | Finish,
    Field(discriminator="type"),
]
EVENT_ADAPTER = TypeAdapter(Event)


def _stream(payload: dict) -> None:
    """Also send the event to LangGraph's custom stream when running inside a graph."""
    try:
        writer = get_stream_writer()
    except RuntimeError:  # called outside a graph run: there is no stream, which is fine
        return
    writer({"event": payload})


class EventLog:
    """Append-only, validated event writer for one run. Safe to reopen when a run resumes."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()
        if self.path.exists():  # resumed run: keep one continuous timeline
            first = self.path.read_text(encoding="utf-8").splitlines()[:1]
            if first:
                self.start = json.loads(first[0])["ts"]

    def emit(self, type: str, **fields) -> dict:
        now = time.time()
        event = EVENT_ADAPTER.validate_python({"type": type, "t": round(now - self.start, 3), "ts": now, **fields})
        payload = event.model_dump(by_alias=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _stream(payload)
        return payload


def read_events(path: Path | str) -> list[dict]:
    """Load and validate every event in a file (raises on the first invalid line)."""
    events = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                events.append(EVENT_ADAPTER.validate_json(line).model_dump(by_alias=True))
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: invalid event: {exc}") from exc
    return events
```

`t` is seconds since the run started. A resumed run reopens the file and keeps the first event's
start time, so a replay plays as one continuous run.

#### From state to an agent's reads

Create `devteam/artifacts.py`. `load_inputs` builds exactly an agent's `reads`: some are computed
from state (`request`, `task_board`, `budget`, ...), the rest are artifact files. `sync_state`
reports what the workspace holds after an agent ran.

```python
"""Bridge between TeamState, the workspace's artifacts and an agent's `reads`.

Agents communicate only through artifacts (structured documents in workspace/.devteam/). Before an
activation, load_inputs() gathers exactly the agent's `reads`; afterwards sync_state() reports what
the workspace now holds so the graph state stays in step.
"""

import json
import shutil
from functools import cache

import yaml

from devteam.config import MAX_ACTIVATIONS, STACKS_FILE
from devteam.team_state import TeamState
from devteam.toolbox import ToolError, Workspace

# A `reads`/`writes` name whose artifact file is named differently.
ALIASES = {"task_board": "tasks.json"}


def artifact_index(ws: Workspace) -> str:
    """One line per artifact: name, size and its first heading, so the Tech Lead can see progress."""
    lines = []
    for name, rel in ws.artifacts().items():
        text = (ws.root / rel).read_text(encoding="utf-8")
        first = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
        lines.append(f"- {name} ({rel.rsplit('/', 1)[-1]}, {len(text.splitlines())} lines): {first[:80]}")
    return "\n".join(lines) or "(no artifacts yet)"


@cache
def stack_catalog() -> str:
    """Stack profiles the architect may pick from: name, language, kind and use_when.

    Non-Python stacks run in Docker; without Docker only Python profiles are offered."""
    data = yaml.safe_load(STACKS_FILE.read_text(encoding="utf-8"))
    docker = shutil.which("docker") is not None
    stacks = [s for s in data["stacks"] if docker or s.get("language") == "python"]
    lines = [f"- {s['name']} ({s.get('language')}, {s.get('kind')}): {s.get('use_when', '')}" for s in stacks]
    note = "" if docker else "\n(Only Python stacks are available: Docker is not installed.)"
    return f"default: {data.get('default_stack')}\n" + "\n".join(lines) + note


def _read(ws: Workspace, name: str) -> str | None:
    try:
        return ws.read_artifact(ALIASES.get(name, name))
    except ToolError:
        return None


def load_inputs(reads: list[str], state: TeamState, ws: Workspace, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return {read_name: text} for an agent's `reads`. Unknown/missing reads are left out, and the
    builder shows them as "(not provided)". `extra` (e.g. the Tech Lead's handoff) wins over files."""
    computed = {
        "request": state["request"],
        "task_board": json.dumps(state["task_board"], indent=1) if state["task_board"] else None,
        "artifact_index": artifact_index(ws),
        "last_agent_result": json.dumps(state["last_result"]) if state["last_result"] else None,
        "budget": f"activations used: {state['activations']} of {MAX_ACTIVATIONS}",
        "stack_catalog": stack_catalog() if "stack_catalog" in reads else None,
    }
    inputs: dict[str, str] = {}
    for name in reads:
        value = (extra or {}).get(name) or computed.get(name) or _read(ws, name)
        if value:
            inputs[name] = value
    return inputs


def sync_state(ws: Workspace) -> dict:
    """State update reflecting the workspace: the artifact index, and the task board if tasks.json parses."""
    update: dict = {"artifacts": ws.artifacts()}
    raw = _read(ws, "task_board")
    if raw:
        try:
            tasks = json.loads(raw)
        except json.JSONDecodeError:
            tasks = None
        if isinstance(tasks, list) and all(isinstance(t, dict) and "id" in t for t in tasks):
            update["task_board"] = tasks
    return update
```

#### JSON artifacts must parse

`write_artifact` validates `.json` artifacts the same way `edit_file` validates Python:

```python
@staticmethod
def _check_json(content: str) -> str:
    """JSON artifacts must parse. Content that is a JSON *string* holding JSON (escaped one level
    too deep, e.g. literal \\n everywhere) is unwrapped; anything else invalid is rejected."""
    try:
        value = json.loads(content)
        if isinstance(value, str):
            value = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ToolError(
            f"rejected: not valid JSON (line {exc.lineno} col {exc.colno}: {exc.msg}); "
            "send the raw JSON document, not an escaped string"
        ) from exc
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"
```

#### The output contract

The builder now refuses to accept "done" for work that doesn't exist. After an agent with
`write_artifact` reports done, every artifact in its `writes` must be in the workspace. If one is
missing, the agent gets one nudge with two extra tool rounds; if it's still missing, the result is
downgraded to `failed`.

```python
        def missing_outputs() -> list[str]:
            """Output contract: an agent with write_artifact must leave every `writes` artifact behind."""
            if "write_artifact" not in by_name:
                return []
            have = workspace.artifacts()
            return [w for w in spec.writes if ALIASES.get(w, w).split(".")[0] not in have]

        result, raw = final_result(tool_loop(ask(model, set(by_name))))
        if result.status == "done" and (absent := missing_outputs()):
            print(f"  [{spec.name}] reported done but did not write {absent}; asking it to write them")
            messages.append(HumanMessage(OUTPUTS_NOTE.format(missing=", ".join(absent))))
            limit = steps + 2  # a small extra budget just for writing the missing outputs
            result, raw = final_result(tool_loop(ask(model, set(by_name))))
            if result.status == "done" and (absent := missing_outputs()):
                result = AgentResult(status="failed", summary=f"Claimed done but never wrote {absent}.")
```

The builder also emits `activate`, `tool_call`, `artifact_written`, `blocked` and `result` events, and
counts tokens from each reply's `usage_metadata`.

#### A resumable pipeline

Create `devteam/team_graph.py` (v1 owns `graph.py`). For now the order is fixed: product_manager,
architect, project_manager, finish. Step 14 replaces the fixed edges with the Tech Lead.

```python
"""v2 team graph with a SQLite checkpointer, so any run can be resumed by its id.

Step 13 wires a FIXED planning pipeline (product_manager -> architect -> project_manager); step 14
replaces the fixed edges with the Tech Lead supervisor. (Named team_graph.py: v1 owns graph.py.)

    python -m devteam.team_graph "build a ..."          # new run
    python -m devteam.team_graph --resume <run_id>      # continue a killed/crashed run
"""

import argparse
import sqlite3
import sys
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from devteam.artifacts import load_inputs, sync_state
from devteam.builder import build_agent
from devteam.config import CHECKPOINT_DB, RECURSION_LIMIT, RUNS_ROOT, WORKSPACE_ROOT
from devteam.events import EventLog
from devteam.registry import load_registry
from devteam.team_state import TeamState, initial_team_state
from devteam.toolbox import Workspace

PIPELINE = ["product_manager", "architect", "project_manager"]


def new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def workspace_for(run_id: str) -> Workspace:
    return Workspace(WORKSPACE_ROOT / run_id)


def events_for(run_id: str) -> EventLog:
    return EventLog(RUNS_ROOT / run_id / "events.jsonl")


def agent_node(name: str):
    """A graph node that runs one registry agent on the current state."""
    registry = load_registry()
    spec = registry.get(name)
    run_agent = build_agent(spec, registry.shared_rules)

    def node(state: TeamState) -> dict:
        ws = workspace_for(state["run_id"])
        out = run_agent(load_inputs(spec.reads, state, ws), ws, events_for(state["run_id"]))
        result = out["result"].model_dump()
        return {
            **sync_state(ws),
            "last_result": {"agent": name, **result},
            "activations": 1,
            "messages": [AIMessage(content=f"{result['status']}: {result['summary']}", name=name)],
        }

    node.__name__ = name
    return node


def finish(state: TeamState) -> dict:
    status = "done" if state["task_board"] else "partial"
    reason = f"{len(state['task_board'])} tasks planned" if state["task_board"] else "no task board produced"
    events_for(state["run_id"]).emit("finish", status=status, reason=reason)
    print(f"[finish] status={status} reason={reason}")
    return {"status": status, "reason": reason}


def build_pipeline(checkpointer=None):
    builder = StateGraph(TeamState)
    previous = START
    for name in PIPELINE:
        builder.add_node(name, agent_node(name))
        builder.add_edge(previous, name)
        previous = name
    builder.add_node("finish", finish)
    builder.add_edge(previous, "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)


def open_checkpointer() -> SqliteSaver:
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False))


def run_config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}


def start(request: str, run_id: str | None = None) -> dict:
    run_id = run_id or new_run_id()
    print(f"[run] id={run_id} workspace={WORKSPACE_ROOT / run_id}")
    graph = build_pipeline(open_checkpointer())
    return graph.invoke(initial_team_state(request, run_id), run_config(run_id))


def resume(run_id: str) -> dict:
    """Continue a run from its last checkpoint (the node that was running when it died reruns)."""
    graph = build_pipeline(open_checkpointer())
    config = run_config(run_id)
    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise SystemExit(f"no checkpoint for run {run_id!r}")
    if not snapshot.next:
        print(f"[resume] run {run_id} already finished (status={snapshot.values['status']})")
        return snapshot.values
    print(f"[resume] run {run_id}: continuing at {list(snapshot.next)}")
    return graph.invoke(None, config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run (or resume) the v2 planning pipeline.")
    parser.add_argument("request", nargs="?", help="what to build")
    parser.add_argument("--run-id", help="id for a new run (default: timestamp)")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue a run from its last checkpoint")
    args = parser.parse_args()
    if args.resume:
        state = resume(args.resume)
    elif args.request:
        state = start(args.request, args.run_id)
    else:
        parser.error("give a request or --resume RUN_ID")
    print(f"[run] status={state['status']} activations={state['activations']} "
          f"artifacts={sorted(state['artifacts'])} tasks={len(state['task_board'])}")
    return 0 if state["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
```

**Design notes**
- **The thread id is the run id.** The checkpointer saves state after every node under
  `thread_id`. `graph.invoke(None, config)` continues from the last saved node; whichever node was
  running when the process died runs again from the start.
- **`check_same_thread=False`** lets the SQLite connection be used from LangGraph's worker threads.
- **The event log is separate from the checkpoint.** The checkpoint is the engine's private state;
  events are a public, append-only record that anything (the office, a dashboard, you) can read.
- **Computed reads.** `stack_catalog` is built from `stacks.yaml`. Without Docker it lists only
  Python profiles, so the architect can't pick a stack that can't run here.

**Check**

```powershell
.venv\Scripts\python -m pytest checks -q
.venv\Scripts\python -X utf8 -m checks.phase02
```

Expected: the run is killed after two agents (`killed=True`), resume continues at
`project_manager`, every agent ran exactly once, the task board is printed, then
`phase 2 check: PASS`. The events are in `runs/<run_id>/events.jsonl`.

**Try it**
- Start a run, press Ctrl+C during the architect, then run
  `python -m devteam.team_graph --resume <run_id>` and watch it pick up at the architect.
- Open `runs/<run_id>/events.jsonl` and follow one agent from `activate` to `result`.

**Check yourself**
1. Why can't `activations` be a plain `int` field if two nodes might run in the same step?
2. Why does a resumed run rerun the node that was interrupted, rather than continuing inside it?
3. What does the output contract catch that the result block alone can't?

**Gotchas**
- **"Blocked" is often correct.** On the first run the architect returned `blocked` because nothing
  supplied its `stack_catalog`. That's the ask-don't-guess rule working; the fix was to provide the
  read, not to change the prompt.
- **Claimed but not done.** The architect once replied "Created design.md" without calling
  `write_artifact`; that's what the output contract now catches.
- **JSON escaped one level too deep.** The project manager sent `tasks.json` with literal `\n`
  sequences. The validator unwraps that case and rejects anything else with the parser's error;
  the agent fixed it on the next try.
- **Killing a run to test resume is a race.** If the next agent is quick, the run can finish before
  the kill. The check polls every 0.1s and kills as soon as the second `result` event appears.

---

### Step 14: Tech Lead supervisor (handoff → Command)

**Goal:** replace the fixed planning chain with a Tech Lead that chooses which planning agent runs
next. For this phase, routing is limited to `product_manager`, `architect`, and `project_manager`.
The graph only exposes the role that owns the next missing artifact, and sends the run to `FINISH`
when planning is complete or a planning agent is blocked.

**Concepts:** [LangGraph `Command`](https://langchain-ai.github.io/langgraph/concepts/low_level/#command),
validated tool calls, structured handoffs, and bounded supervisor loops.

#### The handoff contract

`agents.yaml` gives the Tech Lead the `handoff` tool and tells it to select one target. Each agent
handoff carries four fields: a concrete objective, the artifacts or files to read, the expected
output, and boundaries. The graph exposes only the planning agent responsible for the next missing
artifact; `FINISH` is available when planning is complete or blocked. The tool rejects an unknown
or currently unavailable target, and a worker handoff missing its objective, output or boundaries.
If the model omits the input list, the tool fills it from that agent's configured reads. Workers
receive the handoff as a `handoff` input alongside the artifacts listed in their own `reads` field.

The builder records a successful handoff as an event and returns its validated decision to the
graph node. This keeps routing in the graph layer: the model cannot directly call arbitrary Python
or jump to agents outside this phase.

#### Route with `Command`

`TeamState` now holds the current handoff. The graph starts at `tech_lead`; each planning agent
returns to it after producing its artifact. The Tech Lead node returns a `Command` that both stores
the handoff and names the next node:

```python
if decision["target"] == "FINISH":
    return Command(goto="finish", update={**update, "handoff": None})
return Command(goto=decision["target"], update={**update, "handoff": decision})
```

That creates a supervisor loop instead of the former fixed product-manager → architect → project-
manager edges. The graph checks for completed artifacts before calling the Tech Lead again; this
avoids an extra model call to finish. A run stops as partial if a planner is blocked, the Tech Lead
fails to make a valid route, or the run reaches `MAX_ACTIVATIONS`. The handoff builder accepts only
one route per activation. Normal runs still use the step 13 SQLite checkpointer, so a resumed run
restores both its route and handoff state. After the task board is complete, Step 15 routes to the
approval interrupt before `finish`; blocked planning still finishes as partial.

#### Check the phase

Run the offline checks, including a deterministic graph run, then a live planning run (requires
`.env` and provider quota):

```powershell
.venv\Scripts\python -m pytest checks -q
.venv\Scripts\python -X utf8 -m checks.phase03
```

The deterministic check verifies the graph's `Command` routes without an API call. The live check
expects `product_manager → architect → project_manager → approval`, one run of each planning agent,
an approved done status, and the `prd`, `design`, and `tasks.json` artifacts. It also checks that
each planning handoff is followed by the selected agent with the same objective. The log is saved to
`runs/<run_id>/events.jsonl`; the corresponding artifacts are in `workspace/<run_id>/.devteam/`.

**Try it:** run `.venv\Scripts\python -m devteam.team_graph "Build a tiny URL shortener API"` and
watch the Tech Lead's route and each agent result. Planning pauses for review; approve it with
`--approve-plan <run_id>`. Use `--resume <run_id>` only for a killed or crashed run.

**Check yourself:**

1. Why does the handoff tool validate its target before the node returns `Command`?
2. Which part of the route survives a process restart: the event log, the checkpoint, or both?
3. What should happen if the model returns a result but never calls `handoff`?

**Gotchas:**

- **A result is not a route.** The Tech Lead must call the validated tool; a result block by itself
  ends the run as partial.
- **Keep the phase boundary explicit.** The domain advisors and implementation agents are not
  routable yet. Later steps expand the allowed target set as those teams are built.
- **Events are the replay interface.** This phase records `handoff`, `activate`, and `approval`
  events; the office viewer planned for a later step can animate those events without rerunning the graph.

### Step 14b: Office replay viewer

**Goal:** watch a saved run without making more model calls. The Phaser page is a viewer over
`events.jsonl`; it does not run agents or modify a workspace.

The UI builds the rooms, desks, and avatars from code. It creates avatars only for agents named in
the loaded run, places them in leadership, planning, build, quality, or ops, and routes handoffs over
a grid that avoids desks. Bubbles show assignments, tool calls, results, and approval events.

Open `office/index.html`, choose **Load sample**, then try **Open events.jsonl** with a file from
`runs/<run_id>/events.jsonl`. Use play, pause, speed, and the timeline to replay it. If the browser
blocks loading the sample while opening the file directly, serve the folder locally:

```powershell
cd office
python -m http.server 8000
```

Then open `http://localhost:8000`. Phaser 3.80.1 is pinned in the page and loaded from cdnjs.

**Check the phase:** load `sample-events.jsonl`, scrub across the three planning handoffs, and verify
the Tech Lead walks to each selected agent while the objective appears in a bubble. Load a real
`events.jsonl` from a previous run and confirm only its participating agents appear.

**Gotchas:**

- **Replay is read-only.** A replay shows past events; it cannot resume or change the run.
- **Local file access varies by browser.** Use the file picker for a real event log; serve the folder
  over localhost if the sample fetch is blocked.
- **The scene is generated from events.** An agent absent from the log has no avatar in that replay.

### Step 15: Human approval with interrupt()

**Goal:** pause a completed planning run so a person can review and edit its task board before the
team proceeds. Step 16's builders will only run after the plan is approved.

#### Pause and resume

After `project_manager` writes `tasks.json`, the Tech Lead routes to an `approval` node. The node
uses LangGraph `interrupt()` to surface the request, task board, and artifact path. The SQLite
checkpointer saves the run with status `awaiting_approval`; no model call is needed to approve or
reject it. The event log records the request and the final decision. The office can now stream the
same graph events over WebSocket and show an editable task board in its meeting room.

Start a plan from PowerShell:

```powershell
.venv\Scripts\python -m devteam.team_graph "Build a tiny URL shortener API"
```

When it pauses, review `workspace/<run_id>/.devteam/tasks.json`. You may edit the JSON to add,
remove, reorder, or revise tasks. The approval step checks that the board is a non-empty array with
unique task IDs, titles, and valid dependency IDs. Then resume with either:

```powershell
.venv\Scripts\python -m devteam.team_graph --approve-plan <run_id>
.venv\Scripts\python -m devteam.team_graph --reject-plan <run_id>
```

Approval loads the latest file contents, so edits made before resuming become the graph's task
board. Rejection ends the run as partial. `--resume` remains for killed or crashed runs and does not
bypass a pending approval.

#### Live office

Install the updated requirements, then run the local server from the repository root:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m server
```

Open `http://127.0.0.1:8000`, enter a request, and press **Start run**. The browser receives each
event while the graph is running. At the approval interrupt, the meeting room shows the task board
as editable JSON; approve or reject it there. The local server also writes the normal
`runs/<run_id>/events.jsonl` and SQLite checkpoint, so the same run can be inspected or resumed
from the CLI.

**Try it:** start a live run, edit the task board in the meeting room, and approve it. Start another
run and reject its plan. Then inspect the saved event log and checkpoint state.

**Check yourself:**

1. Why does the approval node read the task file again when it resumes?
2. What keeps `--resume` from accidentally treating a pause as approval?
3. How does the reducer remove a task that a person deleted from the JSON file?

**Gotchas:**

- **The checkpoint stores the pause.** Restarting the process does not restart the LLM planning
  agents; `--approve-plan` resumes the saved interrupt.
- **Approval is explicit.** Editing `tasks.json` alone does not start builders or mark the plan
  approved.
- **The event stream is shared.** Replay reads the saved JSONL log; live mode receives the same
  validated events from LangGraph's custom stream.

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

**Why not let agents use a shell?**
A shell can do anything, including delete things outside the project, and its output is
unbounded. Purpose-built tools refuse what they shouldn't do, explain why, and return just enough
output. Commands that must run (tests, installs) get their own allow-listed tools in later steps.

**Why keep an event log when the checkpointer already saves state?**
The checkpoint holds the latest state, overwritten step by step and shaped for the engine. The event
log holds the whole history in a small public format, so a viewer can replay a run second by second
without importing any engine code.

## Glossary

- **Registry:** the validated list of agent specs loaded from `agents.yaml`.
- **AgentSpec:** one agent's definition: name, team, kind, tier, reads, writes, tools, budget, prompt.
- **Builder:** the function that turns an AgentSpec into a callable agent.
- **Model tier:** an abstract model size (`strong` or `fast`) mapped to a real model in config.
- **Result block:** the JSON every agent ends with: status, summary, questions, handoff_notes.
- **Blocked:** an agent's status when required information is missing; it returns questions instead of guessing.
- **Reads / writes:** the artifacts an agent receives as input and is expected to produce.
- **Shared rules:** instructions prefixed to every LLM agent's prompt.
- **Workspace:** one run's project folder; every agent file operation is locked inside it.
- **Artifact:** a team document (PRD, design, task board, reports) saved in the workspace's `.devteam/` folder.
- **Tool budget (`max_steps`):** the maximum rounds of tool calls in one agent activation.
- **Phantom tool call:** a model's final answer wrapped as a call to a tool that doesn't exist.
- **TeamState:** the v2 graph state shared by every node, with reducers on multi-writer fields.
- **Checkpoint:** a saved copy of the graph state after a node; the basis of resume and human approval.
- **Thread id:** the key a checkpointer stores a run under; in DevTeam it's the run id.
- **Event:** one line of a run's events.jsonl (activate, handoff, tool_call, artifact_written, tests, blocked, result, approval, finish).
- **Output contract:** the rule that an agent's `writes` must exist before its "done" is accepted.
