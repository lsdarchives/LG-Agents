r"""Phase 2 (step 13) live check: kill a run after its 2nd agent, resume it, and it finishes.

Starts the planning pipeline in a subprocess, kills it once two agents have reported, resumes it
from the SQLite checkpoint, then validates the run's events.jsonl.
Run: .venv\Scripts\python -X utf8 -m checks.phase02   (needs .env; ~6-10 LLM calls)
"""

import subprocess
import sys
import time
import uuid
from collections import Counter

from devteam.config import RUNS_ROOT
from devteam.events import read_events
from devteam.team_graph import resume

REQUEST = "A tiny API to shorten URLs: POST a long URL, get a short code; GET the code redirects."


def results(path) -> list[dict]:
    try:
        return [e for e in read_events(path) if e["type"] == "result"]
    except (FileNotFoundError, ValueError):
        return []


def main() -> int:
    run_id = f"check-phase02-{uuid.uuid4().hex[:6]}"
    events_path = RUNS_ROOT / run_id / "events.jsonl"
    log = open(RUNS_ROOT.parent / "workspace" / f"{run_id}.log", "w", encoding="utf-8")  # noqa: SIM115
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-m", "devteam.team_graph", "--run-id", run_id, REQUEST],
        stdout=log, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 600
    while len(results(events_path)) < 2 and proc.poll() is None and time.time() < deadline:
        time.sleep(0.1)
    time.sleep(0.3)  # let the 2nd node's checkpoint be written (it follows the result event)
    killed = proc.poll() is None
    proc.kill()
    proc.wait()
    log.close()
    before = [e["agent"] for e in results(events_path)]
    print(f"killed={killed} after agents: {before}")

    state = resume(run_id)
    events = read_events(events_path)  # raises if any line is invalid
    ran = Counter(e["agent"] for e in events if e["type"] == "result")
    problems = []
    if not killed:
        problems.append("the run ended before it could be killed")
    if ran["product_manager"] != 1:
        problems.append(f"product_manager ran {ran['product_manager']} times (resume should not rerun it)")
    if ran["project_manager"] != 1:
        problems.append("project_manager did not run exactly once after resume")
    if not any(e["type"] == "finish" for e in events):
        problems.append("no finish event")
    if state["status"] != "done":
        problems.append(f"final status={state['status']} ({state['reason']})")

    print(f"events: {len(events)} valid; results per agent: {dict(ran)}")
    print(f"final: status={state['status']} tasks={len(state['task_board'])} artifacts={sorted(state['artifacts'])}")
    for task in state["task_board"][:5]:
        print(f"  {task.get('id')}: {task.get('title')} [{task.get('owner_role')}]")
    print("\nphase 2 check:", "PASS" if not problems else f"FAIL: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
