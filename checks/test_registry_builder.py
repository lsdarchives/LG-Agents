"""Step 11 offline checks: registry validation, result parsing, input isolation."""

import pytest
from pydantic import ValidationError

from devteam.builder import AgentOutputError, parse_result, render_inputs
from devteam.registry import AgentSpec, Registry, load_registry

BASE = {"name": "x_agent", "team": "build", "kind": "llm", "description": "does a thing well",
            "model_tier": "fast", "max_steps": 3, "prompt": "p"}


def test_real_registry_loads():
    registry = load_registry()
    assert len(registry.agents) == 32
    assert registry.get("tech_lead").model_tier == "strong"


@pytest.mark.parametrize("change", [
    {"tools": ["view_files"]},      # unknown tool
    {"team": "sales"},              # unknown team
    {"prompt": ""},                 # llm agent without prompt
    {"max_step": 3},                # typo -> extra field
    {"kind": "tool"},               # tool agent must not have tier/prompt
])
def test_bad_specs_rejected(change):
    with pytest.raises(ValidationError):
        AgentSpec(**{**BASE, **change})


def test_duplicate_names_rejected():
    with pytest.raises(ValidationError):
        Registry(shared_rules="x" * 30, agents=[BASE, BASE])


def test_last_status_block_wins():
    text = 'review {"verdict":"fail","findings":[]} ```json\n{"status":"done","summary":"ok"}\n```'
    result = parse_result(text)
    assert (result.status, result.summary) == ("done", "ok")


@pytest.mark.parametrize("text", ["no json", '{"status":"maybe","summary":"x"}'])
def test_bad_result_rejected(text):
    with pytest.raises(AgentOutputError):
        parse_result(text)


def test_inputs_limited_to_reads():
    spec = AgentSpec(**{**BASE, "reads": ["prd", "design"]})
    rendered = render_inputs(spec, {"prd": "P", "secret": "S"})
    assert rendered == "## prd\nP\n\n## design\n(not provided)"
