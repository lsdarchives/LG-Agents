r"""Phase 1 (step 12) live check: agents use their tools to write real artifacts.

product_manager writes the PRD with write_artifact; architect reads it from the workspace and writes
design.md. Run: .venv\Scripts\python -X utf8 -m checks.phase01   (needs .env)
"""

import sys
from pathlib import Path

from devteam.builder import get_agent
from devteam.toolbox import Workspace

REQUEST = "A tiny API to shorten URLs: POST a long URL, get a short code; GET the code redirects."
STACKS = "python-fastapi: use_when small JSON APIs; pytest + httpx TestClient; SQLite ok."


def main() -> int:
    ws = Workspace(Path("workspace") / "check-phase01")
    for name in ("prd", "design"):
        (ws.root / ".devteam" / f"{name}.md").unlink(missing_ok=True)

    problems = []
    pm = get_agent("product_manager")({"request": REQUEST}, ws)
    if not any(c["tool"] == "write_artifact" for c in pm["tool_calls"]):
        problems.append("product_manager never called write_artifact")
    if "prd" not in ws.artifacts():
        problems.append("no prd artifact saved")
    else:
        arch = get_agent("architect")({"prd": ws.read_artifact("prd"), "stack_catalog": STACKS}, ws)
        if arch["result"].status != "done":
            problems.append(f"architect status={arch['result'].status}")
        if "design" not in ws.artifacts():
            problems.append("no design artifact saved")

    print("\nartifacts:", ws.artifacts())
    for name in ws.artifacts():
        text = ws.read_artifact(name)
        print(f"--- {name}: {len(text.splitlines())} lines ---\n" + "\n".join(text.splitlines()[:12]))
    print("\nphase 1 check:", "PASS" if not problems else f"FAIL: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
