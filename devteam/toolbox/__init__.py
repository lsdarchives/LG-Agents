"""v2 agent tools. Named `toolbox` so it doesn't shadow v1's devteam/tools.py.

make_tools(workspace, names) turns tool names from agents.yaml into LangChain tools bound to one
workspace. A refused request (ToolError) or bad arguments come back to the agent as an
"ERROR: ..." reply it can read and react to, instead of crashing the run.
"""

from langchain_core.tools import BaseTool, StructuredTool

from devteam.toolbox.workspace import ToolError, Workspace


def _error_text(exc: Exception) -> str:
    return f"ERROR: {exc}"


def _bad_args_text(exc: Exception) -> str:
    return f"ERROR: invalid arguments: {exc}"


def workspace_tools(ws: Workspace) -> dict[str, BaseTool]:
    """All workspace tools, bound to `ws`, keyed by the names agents.yaml uses."""

    def list_tree(depth: int = 3) -> str:
        """Show the project's file tree (directories first), `depth` levels deep."""
        return ws.list_tree(depth)

    def view_file(path: str, start: int = 1) -> str:
        """Show 100 numbered lines of a project file, starting at line `start`."""
        return ws.view_file(path, start)

    def search_files(pattern: str) -> str:
        """List project files containing `pattern` (plain text, case-insensitive) with match counts."""
        return ws.search_files(pattern)

    def create_file(path: str, content: str) -> str:
        """Create a NEW project file with `content`. Fails if it exists (then use edit_file)."""
        return ws.create_file(path, content)

    def edit_file(path: str, start_line: int, end_line: int, new_text: str) -> str:
        """Replace lines start_line..end_line (1-based, inclusive) of a file with new_text.
        To insert without deleting use end_line = start_line - 1. Python edits that break syntax are rejected."""
        return ws.edit_file(path, start_line, end_line, new_text)

    def write_artifact(name: str, content: str) -> str:
        """Save a team document by name (e.g. 'prd', 'design', 'tasks.json'); overwrites the old version."""
        return ws.write_artifact(name, content)

    def read_artifact(name: str) -> str:
        """Read a team document by name (e.g. 'prd', 'design')."""
        return ws.read_artifact(name)

    fns = [list_tree, view_file, search_files, create_file, edit_file, write_artifact, read_artifact]
    return {
        fn.__name__: StructuredTool.from_function(
            func=fn,
            name=fn.__name__,
            description=fn.__doc__,
            handle_tool_error=_error_text,
            handle_validation_error=_bad_args_text,
        )
        for fn in fns
    }


def make_tools(ws: Workspace, names: list[str]) -> tuple[list[BaseTool], list[str]]:
    """Return (tools for `names` that exist, names not implemented yet)."""
    available = workspace_tools(ws)
    tools = [available[n] for n in names if n in available]
    missing = [n for n in names if n not in available]
    return tools, missing


__all__ = ["ToolError", "Workspace", "make_tools", "workspace_tools"]
