"""Run one generated Python task's tests and lint in a temporary virtual environment."""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
import threading
import venv
from pathlib import Path

from devteam.toolbox import Workspace

COMMAND_TIMEOUT = 120
MAX_REPORT_CHARS = 5000
_ENVIRONMENTS: dict[str, "_ProjectEnvironment"] = {}
_ENVIRONMENTS_LOCK = threading.Lock()


def task_test_path(task_id: str) -> str:
    """Stable test path shared by test_engineer and qa_runner."""
    slug = re.sub(r"[^a-z0-9]+", "_", task_id.lower()).strip("_")
    if not slug:
        raise ValueError("task id must contain letters or digits")
    return f"tests/test_{slug}.py"


def _fingerprint(workspace: Path) -> tuple[str, list[str]]:
    requirements = workspace / "requirements.txt"
    pyproject = workspace / "pyproject.toml"
    manifest = requirements if requirements.is_file() else pyproject if pyproject.is_file() else None
    contents = manifest.read_bytes() if manifest else b""
    return hashlib.sha256(contents).hexdigest(), ([str(manifest)] if manifest else [])


class _ProjectEnvironment:
    """A cached temporary venv for one run workspace; project dependencies install once per manifest."""

    def __init__(self, workspace: Path):
        self.temp = tempfile.TemporaryDirectory(prefix="devteam-qa-")
        self.root = Path(self.temp.name)
        self.python = self.root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        venv.EnvBuilder(with_pip=True).create(self.root / "venv")
        self.fingerprint: str | None = None
        self.workspace = workspace

    def install(self) -> tuple[bool, str]:
        fingerprint, manifests = _fingerprint(self.workspace)
        if fingerprint == self.fingerprint:
            return True, "Dependencies already installed for this workspace."
        command = [str(self.python), "-m", "pip", "install", "--disable-pip-version-check", "pytest", "ruff"]
        if manifests and manifests[0].endswith("requirements.txt"):
            command.extend(["-r", manifests[0]])
        elif manifests:
            command.extend(["-e", str(self.workspace)])
        code, output = _run(command, self.workspace, timeout=COMMAND_TIMEOUT)
        if code == 0:
            self.fingerprint = fingerprint
        return code == 0, output


def _environment(workspace: Path) -> _ProjectEnvironment:
    key = str(workspace.resolve())
    with _ENVIRONMENTS_LOCK:
        environment = _ENVIRONMENTS.get(key)
        if environment is None:
            environment = _ProjectEnvironment(workspace)
            _ENVIRONMENTS[key] = environment
        return environment


def _run(command: list[str], cwd: Path, timeout: int = COMMAND_TIMEOUT) -> tuple[int | None, str]:
    env = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONUTF8": "1",
        "PYTHONPATH": str(cwd) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or b"") + (exc.stderr or b"")
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return None, f"TIMEOUT after {timeout}s\n{output}"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _count(pattern: str, output: str) -> int:
    match = re.search(pattern, output)
    return int(match.group(1)) if match else 0


def _trim(output: str) -> str:
    return output if len(output) <= MAX_REPORT_CHARS else output[-MAX_REPORT_CHARS:]


def run_task_checks(workspace: Workspace, task_id: str, test_file: str | None = None) -> dict:
    """Run ruff autofix, one task's pytest file, then a clean ruff pass."""
    target = test_file or task_test_path(task_id)
    try:
        target_path = workspace.resolve(target)
        if not target_path.is_file():
            raise ValueError(f"task test file {target!r} does not exist")
        environment = _environment(workspace.root)
        installed, install_output = environment.install()
        if not installed:
            return {
                "task_id": task_id,
                "stack": "python",
                "passed": 0,
                "failed": 1,
                "lint_issues": 0,
                "tests_clean": False,
                "lint_clean": False,
                "output": _trim(f"Dependency installation failed:\n{install_output}"),
            }

        fixer_code, fixer_output = _run(
            [str(environment.python), "-m", "ruff", "check", "--fix", "."], workspace.root
        )
        pytest_code, pytest_output = _run(
            [str(environment.python), "-m", "pytest", "-q", "--tb=short", target], workspace.root
        )
        lint_code, lint_output = _run(
            [str(environment.python), "-m", "ruff", "check", "."], workspace.root
        )
        passed = _count(r"(\d+) passed", pytest_output)
        failed = _count(r"(\d+) failed", pytest_output) + _count(r"(\d+) error", pytest_output)
        tests_clean = pytest_code == 0 and passed > 0
        if pytest_code != 0 and failed == 0:
            failed = 1
        lint_issues = _count(r"Found (\d+) errors?", lint_output)
        if lint_code != 0 and lint_issues == 0:
            lint_issues = 1
        output = "\n\n".join(
            section
            for section in (
                f"ruff --fix (exit {fixer_code}):\n{fixer_output}" if fixer_output else "ruff --fix completed.",
                f"pytest {target} (exit {pytest_code}):\n{pytest_output}",
                f"ruff check (exit {lint_code}):\n{lint_output}" if lint_output else "ruff check clean.",
            )
        )
        return {
            "task_id": task_id,
            "stack": "python",
            "passed": passed,
            "failed": failed,
            "lint_issues": lint_issues,
            "tests_clean": tests_clean,
            "lint_clean": lint_code == 0,
            "output": _trim(output),
        }
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        return {
            "task_id": task_id,
            "stack": "python",
            "passed": 0,
            "failed": 1,
            "lint_issues": 0,
            "tests_clean": False,
            "lint_clean": False,
            "output": f"Quality runner error: {exc}",
        }
