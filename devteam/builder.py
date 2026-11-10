"""Turn an AgentSpec from agents.yaml into a callable agent.

An llm agent is a small tool-calling loop: the model either calls tools (bound to one workspace)
or gives its final answer. The loop is capped at the agent's `max_steps` tool rounds, and the
final reply must end with the shared JSON result block.
"""

import json
from collections.abc import Callable
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from devteam.artifacts import ALIASES
from devteam.events import EventLog
from devteam.llm import get_llm, invoke_messages
from devteam.registry import AgentSpec, load_registry
from devteam.toolbox import Workspace, make_tools

MISSING = "(not provided)"
BUDGET_NOTE = (
    "Your tool budget for this activation is used up. Do not call any more tools. "
    "Reply now with your final answer, ending with the result block."
)
OUTPUTS_NOTE = (
    "You reported done, but these outputs were never saved under their required artifact names: "
    "{missing}. Nothing exists until write_artifact saves each exact name. For .json outputs, pass "
    "raw valid JSON with no Markdown fences. Call write_artifact for each missing output now, then "
    "reply with the result block."
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


def build_agent(
    spec: AgentSpec,
    shared_rules: str,
    handoff_targets: dict[str, list[str]] | list[str] | None = None,
    task_id: str | None = None,
) -> Callable[..., dict]:
    """Return agent(inputs, workspace, events=None, objective=None)
    -> {"result", "raw", "steps", "tool_calls", "tokens"}."""
    if spec.kind == "tool":
        raise NotImplementedError(f"{spec.name!r} is a tool node; tool nodes are built in later steps")
    system = system_prompt(spec, shared_rules)

    def agent(
        inputs: dict[str, str],
        workspace: Workspace,
        events: EventLog | None = None,
        objective: str | None = None,
    ) -> dict:
        tools, missing = make_tools(workspace, spec.tools, handoff_targets, task_id=task_id)
        by_name = {t.name: t for t in tools}
        plain = get_llm(spec.model_tier)
        model = plain.bind_tools(tools) if tools else plain
        emit = events.emit if events else (lambda *a, **k: None)
        print(f"[{spec.name}] tier={spec.model_tier} reads={spec.reads} tools={list(by_name)}"
              + (f" (not built yet: {missing})" if missing else ""))
        emit("activate", agent=spec.name, objective=objective or spec.description, reads=list(inputs))

        messages = [SystemMessage(system), HumanMessage(render_inputs(spec, inputs))]
        tool_calls: list[dict] = []
        routing_decision: dict | None = None
        steps = 0
        tokens = 0

        def ask(chat_model, names: set[str] | None = None):
            nonlocal tokens
            reply = invoke_messages(chat_model, messages, spec.model_tier, names)
            tokens += (getattr(reply, "usage_metadata", None) or {}).get("total_tokens", 0)
            messages.append(reply)
            return reply

        limit = spec.max_steps

        def tool_loop(reply):
            """Run tool rounds until the model answers without tools or the budget is used up."""
            nonlocal steps, routing_decision
            while reply.tool_calls:
                if steps >= limit:
                    for call in reply.tool_calls:  # every tool call needs an answer before the next turn
                        messages.append(ToolMessage("ERROR: tool budget used up", tool_call_id=call["id"]))
                    messages.append(HumanMessage(BUDGET_NOTE))
                    return ask(plain)
                steps += 1
                accepted_route = False
                for call in reply.tool_calls:
                    tool = by_name.get(call["name"])
                    if call["name"] == "handoff" and routing_decision is not None:
                        out = ToolMessage(
                            "ERROR: one handoff was already accepted for this activation",
                            tool_call_id=call["id"],
                        )
                    elif tool is None:
                        out = ToolMessage(
                            f"ERROR: unknown tool {call['name']!r}; you have {list(by_name)}",
                            tool_call_id=call["id"],
                        )
                    else:
                        out = tool.invoke(call)  # returns a ToolMessage; errors come back as "ERROR: ..."
                    tool_calls.append({"tool": call["name"], "args": call["args"]})
                    first = str(out.content).splitlines()[0] if out.content else ""
                    print(f"  [{spec.name}] {call['name']}({_short(call['args'])}) -> {first[:90]}")
                    emit("tool_call", agent=spec.name, tool=call["name"], arg=_short(call["args"], 60))
                    if call["name"] == "write_artifact" and first.startswith("Saved artifact"):
                        emit("artifact_written", agent=spec.name, name=str(call["args"].get("name", "")))
                    if call["name"] == "handoff":
                        try:
                            decision = json.loads(str(out.content))
                        except (TypeError, json.JSONDecodeError):
                            decision = {}
                        if decision.get("accepted"):
                            routing_decision = {k: v for k, v in decision.items() if k != "accepted"}
                            accepted_route = True
                            target = "finish" if decision["target"] == "FINISH" else decision["target"]
                            emit(
                                "handoff",
                                **{"from": spec.name},
                                to=target,
                                objective=decision["objective"],
                            )
                    messages.append(out)
                if accepted_route:
                    return AIMessage(
                        content=json.dumps(
                            {
                                "status": "done",
                                "summary": (
                                    f"Routed to {routing_decision['target']}: "
                                    f"{routing_decision['objective']}"
                                ),
                                "questions": [],
                                "handoff_notes": [],
                            }
                        )
                    )
                reply = ask(model, set(by_name))
            return reply

        def final_result(reply) -> tuple[AgentResult, str]:
            try:
                return parse_result(reply.text), reply.text
            except AgentOutputError:
                messages.append(HumanMessage(REPAIR_NOTE))  # one repair attempt, then give up
                raw = ask(plain).text
                return parse_result(raw), raw

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
        print(f"[{spec.name}] status={result.status} steps={steps} tokens={tokens} "
              f"summary={result.summary[:100]!r}")
        if result.status == "blocked":
            emit("blocked", agent=spec.name, questions=result.questions)
        emit("result", agent=spec.name, status=result.status, summary=result.summary, steps=steps, tokens=tokens)
        return {
            "result": result,
            "raw": raw,
            "steps": steps,
            "tool_calls": tool_calls,
            "tokens": tokens,
            "handoff": routing_decision,
        }

    agent.__name__ = spec.name
    return agent


def get_agent(name: str) -> Callable[..., dict]:
    """Convenience: look up `name` in agents.yaml and build it."""
    registry = load_registry()
    return build_agent(registry.get(name), registry.shared_rules)
