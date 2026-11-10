"""Offline check: a final answer sent as a call to a non-existent tool is recovered as text."""

import json

from devteam.builder import parse_result
from devteam.llm import _recover_phantom_tool_call


class FakeGroqError(Exception):
    def __init__(self, name):
        gen = json.dumps({"name": name, "arguments": {"status": "done", "summary": "ok"}})
        self.body = {"error": {"code": "tool_use_failed", "failed_generation": gen}}


def test_phantom_tool_recovered_as_text():
    msg = _recover_phantom_tool_call(FakeGroqError("json"), {"write_artifact"})
    assert json.loads(msg.content) == {"status": "done", "summary": "ok"}
    assert not msg.tool_calls


def test_unparsable_phantom_keeps_raw_text():
    exc = FakeGroqError("json")
    exc.body["error"]["failed_generation"] = (
        '{"name": "json", "arguments": {"root_cause": "x"}\n{"status": "done", "summary": "ok"}}'
    )
    msg = _recover_phantom_tool_call(exc, {"view_file"})
    assert parse_result(msg.content).summary == "ok"


def test_real_tool_failure_not_recovered():
    assert _recover_phantom_tool_call(FakeGroqError("write_artifact"), {"write_artifact"}) is None


def test_other_errors_not_recovered():
    assert _recover_phantom_tool_call(ValueError("boom"), set()) is None
