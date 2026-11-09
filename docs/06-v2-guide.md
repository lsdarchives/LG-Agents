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

**Why not let agents use a shell?**
A shell can do anything, including delete things outside the project, and its output is
unbounded. Purpose-built tools refuse what they shouldn't do, explain why, and return just enough
output. Commands that must run (tests, installs) get their own allow-listed tools in later steps.

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
