"""Workspace tools: the file operations agents use, all locked inside one run's folder.

Design (agent-computer interface): small outputs, line numbers, clear errors returned as text
(never raised at the agent), and edits that can't leave a Python file unparsable.
"""

import ast
import json
import re
from pathlib import Path

from langchain_core.tools import ToolException

VIEW_LINES = 100  # lines per view_file page
MAX_OUTPUT_CHARS = 6000  # cap on any single tool reply
ARTIFACT_DIR = ".devteam"  # team documents (prd, design, ...) live here, not in the project tree
SKIP_DIRS = {ARTIFACT_DIR, ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
ARTIFACT_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
NO_OUTPUT = "Command ran successfully with no output."


class ToolError(ToolException):
    """A tool refused the request. LangChain turns it into a reply the agent sees, not a crash."""


def _trim(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text or NO_OUTPUT
    return text[:MAX_OUTPUT_CHARS] + f"\n... [trimmed {len(text) - MAX_OUTPUT_CHARS} chars]"


class Workspace:
    """One run's project folder. Every path an agent gives is resolved and checked against it."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ARTIFACT_DIR).mkdir(exist_ok=True)

    # ── path guard ────────────────────────────────────────────────────────────
    def resolve(self, rel: str) -> Path:
        """Return the absolute path for `rel`, or raise ToolError if it escapes the workspace."""
        if not rel or not rel.strip():
            raise ToolError("path is empty")
        path = (self.root / rel.strip()).resolve()
        if not path.is_relative_to(self.root):
            raise ToolError(f"path {rel!r} is outside the workspace; use paths relative to the project root")
        if ARTIFACT_DIR in path.relative_to(self.root).parts:
            raise ToolError(f"{ARTIFACT_DIR}/ holds team artifacts; use read_artifact / write_artifact")
        return path

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # ── read tools ────────────────────────────────────────────────────────────
    def list_tree(self, depth: int = 3) -> str:
        """Compact file tree, directories first, skipping caches, venvs and artifacts."""
        depth = max(1, min(int(depth), 6))
        lines: list[str] = []

        def walk(folder: Path, level: int) -> None:
            entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            for entry in entries:
                if entry.is_dir() and entry.name in SKIP_DIRS:
                    continue
                lines.append("  " * level + entry.name + ("/" if entry.is_dir() else ""))
                if entry.is_dir() and level + 1 < depth:
                    walk(entry, level + 1)

        walk(self.root, 0)
        return _trim("\n".join(lines)) if lines else "(workspace is empty)"

    def view_file(self, path: str, start: int = 1) -> str:
        """Show VIEW_LINES numbered lines from `start`, with the path and total line count."""
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"{path!r} is not a file; call list_tree to see what exists")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        start = max(1, int(start))
        if total and start > total:
            raise ToolError(f"start={start} is past the end; {path} has {total} lines")
        end = min(start + VIEW_LINES - 1, total)
        body = "\n".join(f"{n:>4} | {lines[n - 1]}" for n in range(start, end + 1))
        more = f"\n... {total - end} more lines; call view_file({path!r}, start={end + 1})" if end < total else ""
        return _trim(f"[{self.rel(target)}: lines {start}-{end} of {total}]\n{body}{more}")

    def search_files(self, pattern: str) -> str:
        """List files containing `pattern` (plain text, case-insensitive) with match counts."""
        if not pattern:
            raise ToolError("pattern is empty")
        needle = pattern.lower()
        hits: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(self.root).parts):
                continue
            try:
                count = path.read_text(encoding="utf-8").lower().count(needle)
            except (UnicodeDecodeError, OSError):
                continue
            if count:
                hits.append(f"{self.rel(path)} ({count} matches)")
        if not hits:
            return f"No files contain {pattern!r}."
        return _trim(f"{len(hits)} files contain {pattern!r}:\n" + "\n".join(hits))

    # ── write tools ───────────────────────────────────────────────────────────
    @staticmethod
    def _check_python(path: Path, text: str) -> None:
        if path.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as exc:
                raise ToolError(
                    f"rejected: the result would not parse (line {exc.lineno}: {exc.msg}); file unchanged"
                ) from exc

    def create_file(self, path: str, content: str) -> str:
        """Create a new file. Refuses to overwrite: existing files are changed with edit_file."""
        target = self.resolve(path)
        if target.exists():
            raise ToolError(f"{path!r} already exists; view it and use edit_file to change it")
        self._check_python(target, content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return f"Created {self.rel(target)} ({len(content.splitlines())} lines)."

    def edit_file(self, path: str, start_line: int, end_line: int, new_text: str) -> str:
        """Replace lines start_line..end_line (1-based, inclusive) with new_text.

        To insert without deleting, use end_line = start_line - 1. Python files are syntax-checked
        and the edit is rejected (file unchanged) if the result doesn't parse.
        """
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"{path!r} does not exist; use create_file")
        lines = target.read_text(encoding="utf-8").splitlines()
        start_line, end_line = int(start_line), int(end_line)
        if start_line < 1 or start_line > len(lines) + 1 or end_line < start_line - 1 or end_line > len(lines):
            raise ToolError(f"line range {start_line}-{end_line} is invalid; {path} has {len(lines)} lines")
        replacement = new_text.splitlines()
        updated = lines[: start_line - 1] + replacement + lines[end_line:]
        text = "\n".join(updated) + "\n"
        self._check_python(target, text)
        target.write_text(text, encoding="utf-8", newline="\n")
        # Show the edited region with a little context so the agent can verify the change.
        lo = max(1, start_line - 2)
        hi = min(len(updated), start_line + len(replacement) + 1)
        region = "\n".join(f"{n:>4} | {updated[n - 1]}" for n in range(lo, hi + 1))
        return _trim(f"Edited {self.rel(target)}; lines {lo}-{hi} now:\n{region}")

    # ── artifacts ─────────────────────────────────────────────────────────────
    def _artifact_path(self, name: str) -> Path:
        name = name.strip().lower()
        if not ARTIFACT_NAME.match(name):
            raise ToolError(f"artifact name {name!r} must be lowercase letters, digits, '_', '-' or '.'")
        if "." not in name:
            name += ".md"
        return self.root / ARTIFACT_DIR / name

    @staticmethod
    def _check_json(content: str) -> str:
        """JSON artifacts must parse. Content that is a JSON *string* holding JSON (escaped one level
        too deep, e.g. literal \\n everywhere) is unwrapped; anything else invalid is rejected."""
        try:
            value = json.loads(content)
            if isinstance(value, str):
                value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ToolError(
                f"rejected: not valid JSON (line {exc.lineno} col {exc.colno}: {exc.msg}); "
                "send the raw JSON document, not an escaped string"
            ) from exc
        return json.dumps(value, indent=2, ensure_ascii=False) + "\n"

    def write_artifact(self, name: str, content: str) -> str:
        """Save a team document (e.g. 'prd', 'design', 'tasks.json'). Overwrites the previous version."""
        if not content.strip():
            raise ToolError("artifact content is empty")
        path = self._artifact_path(name)
        if path.suffix == ".json":
            content = self._check_json(content)
        path.write_text(content, encoding="utf-8", newline="\n")
        return f"Saved artifact {path.name} ({len(content.splitlines())} lines)."

    def read_artifact(self, name: str) -> str:
        """Return a team document's full text."""
        path = self._artifact_path(name)
        if not path.exists():
            have = sorted(p.name for p in (self.root / ARTIFACT_DIR).iterdir())
            raise ToolError(f"no artifact {path.name!r}; existing: {have or 'none'}")
        return _trim(path.read_text(encoding="utf-8"))

    def artifacts(self) -> dict[str, str]:
        """name -> relative path of every saved artifact (for the artifact index)."""
        folder = self.root / ARTIFACT_DIR
        return {p.stem: self.rel(p) for p in sorted(folder.iterdir()) if p.is_file()}
