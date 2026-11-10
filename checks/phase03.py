r"""Phase 3 live check: the Tech Lead plans a project and resumes after human approval.

Run: .venv\Scripts\python -X utf8 -m checks.phase03   (needs .env; about 8 model calls)
"""

import sys
import uuid
from collections import Counter

from devteam.events import read_events
from devteam.team_graph import PLANNING_AGENTS, RUNS_ROOT, respond_to_plan, start

REQUEST = "A tiny API to shorten URLs: POST a long URL, get a short code; GET the code redirects."
EXPECTED_ROUTES = [*PLANNING_AGENTS, "approval"]


def main() -> int:
    run_id = f"check-phase03-{uuid.uuid4().hex[:6]}"
    state = start(REQUEST, run_id)
    if state["status"] == "awaiting_approval":
        state = respond_to_plan(run_id, "approve")
    events = read_events(RUNS_ROOT / run_id / "events.jsonl")
    problems = []

    handoffs = [(index, event) for index, event in enumerate(events) if event["type"] == "handoff"]
    routes = [event["to"] for _, event in handoffs]
    if routes != EXPECTED_ROUTES:
        problems.append(f"routes were {routes}, expected {EXPECTED_ROUTES}")
    for index, handoff in handoffs:
        if handoff["from"] != "tech_lead":
            problems.append(f"handoff came from {handoff['from']!r}, not tech_lead")
        if handoff["to"] in PLANNING_AGENTS:
            next_agent = next(
                (event for event in events[index + 1 :] if event["type"] == "activate"),
                None,
            )
            if not next_agent or next_agent["agent"] != handoff["to"]:
                problems.append(f"handoff to {handoff['to']} was not followed by that agent")
            elif next_agent["objective"] != handoff["objective"]:
                problems.append(f"objective was not carried to {handoff['to']}")

    results = Counter(event["agent"] for event in events if event["type"] == "result")
    for name in PLANNING_AGENTS:
        if results[name] != 1:
            problems.append(f"{name} ran {results[name]} times; expected exactly once")
    if not {"prd", "design", "tasks"} <= set(state["artifacts"]):
        problems.append(f"missing planning artifacts: {sorted(state['artifacts'])}")
    if state["status"] != "done":
        problems.append(f"final status={state['status']} ({state['reason']})")
    if not state["plan_approved"]:
        problems.append("the generated plan was not marked approved")
    if not events or events[-1]["type"] != "finish":
        problems.append("event log does not end with finish")

    print(f"routes: {routes}; results: {dict(results)}")
    print(f"artifacts: {sorted(state['artifacts'])}; status: {state['status']}")
    print("phase 3 check:", "PASS" if not problems else f"FAIL: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
