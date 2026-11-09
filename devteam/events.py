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
