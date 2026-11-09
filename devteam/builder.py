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
