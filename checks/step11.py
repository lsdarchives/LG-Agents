r"""Step 11 live check: three agents (two tiers, two teams) reply with a valid result block.

Run: .venv\Scripts\python -X utf8 -m checks.step11   (needs .env)
"""

import json
import tempfile

from devteam.builder import AgentOutputError, get_agent
from devteam.toolbox import Workspace

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


def main() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(tmp)
        for agent_name, agent_inputs in CHECK_INPUTS.items():
            try:
                out = get_agent(agent_name)(agent_inputs, ws)
                print(json.dumps(out["result"].model_dump(), indent=2)[:600], "\n")
            except AgentOutputError as exc:
                failures += 1
                print(f"[{agent_name}] FAILED: {exc}\n")
    print("step 11 check:", "PASS" if failures == 0 else f"{failures} agent(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
