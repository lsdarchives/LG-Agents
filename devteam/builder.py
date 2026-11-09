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
