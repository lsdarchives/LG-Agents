"""Tool functions (no LLM): run pytest and ruff on generated code in a temp dir.

Not real isolation - generated code runs with your user's permissions.
Only run tasks you wrote yourself.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

TIMEOUT_SECONDS = 30


class TestResult(NamedTuple):
    output: str  # pytest output, or "TIMEOUT ..." if it ran too long
    passed: bool  # True only if every test passed (exit code 0)
    score: int  # number of passed tests, parsed from the summary line


class LintResult(NamedTuple):
    output: str  # ruff output
    clean: bool  # True if ruff found no problems


def _run(cmd: list[str], cwd: Path) -> tuple[int | None, str]:
    """Run a command; return (exit code, combined output). Exit code None = timeout."""
    try:
        proc = subprocess.run(
            cmd,
            check=False,  # non-zero exit = failing tests/lint, which is normal here
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT: command took longer than {TIMEOUT_SECONDS}s"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def count_passed(output: str) -> int:
    """Parse e.g. '3 failed, 5 passed in 0.12s' -> 5."""
    match = re.search(r"(\d+) passed", output)
    return int(match.group(1)) if match else 0


def run_tests(code: str, tests: str) -> TestResult:
    """Write solution.py + test_solution.py to a fresh temp dir and run pytest."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        (workdir / "test_solution.py").write_text(tests, encoding="utf-8")
        cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"]
        returncode, output = _run(cmd, workdir)

    if returncode is None:
        return TestResult(output=output, passed=False, score=0)
    # Exit code 0 = all passed. (5 = no tests collected, which also counts as failure.)
    return TestResult(output=output, passed=returncode == 0, score=count_passed(output))


def run_lint(code: str) -> LintResult:
    """Run `ruff check solution.py` in a fresh temp dir, ignoring any project config."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        cmd = [sys.executable, "-m", "ruff", "check", "--isolated", "--no-cache", "solution.py"]
        returncode, output = _run(cmd, workdir)

    return LintResult(output=output, clean=returncode == 0)
