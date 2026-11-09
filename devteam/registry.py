"""Load and validate agents.yaml. A bad entry fails at load time, not mid-run."""

from collections import Counter
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from devteam.config import AGENTS_FILE

# Every tool name an agent may list. Implementations arrive in step 12+; listing them here
# means a typo like "view_files" is caught now instead of when the agent first runs.
KNOWN_TOOLS = frozenset(
    {
        "list_tree",
        "view_file",
        "search_files",
        "create_file",
        "edit_file",
        "run_tests",
        "run_command",
        "write_artifact",
        "read_artifact",
        "handoff",
    }
)


class AgentSpec(BaseModel):
    """One entry under `agents:` in agents.yaml."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    team: Literal["leadership", "domain", "build", "quality", "ops"]
    kind: Literal["llm", "tool"]
    description: str = Field(min_length=10)
    reads: list[str] = []
    writes: list[str] = []
    model_tier: Literal["strong", "fast"] | None = None
    tools: list[str] = []
    max_steps: int | None = Field(default=None, ge=1, le=50)
    prompt: str = ""

    @model_validator(mode="after")
    def _check_kind(self) -> "AgentSpec":
        if self.kind == "llm":
            missing = [f for f in ("model_tier", "max_steps") if getattr(self, f) is None]
            if not self.prompt.strip():
                missing.append("prompt")
            if missing:
                raise ValueError(f"llm agent {self.name!r} is missing {missing}")
        elif self.model_tier or self.prompt or self.tools:
            raise ValueError(f"tool agent {self.name!r} must not set model_tier, prompt or tools")
        unknown = sorted(set(self.tools) - KNOWN_TOOLS)
        if unknown:
            raise ValueError(f"agent {self.name!r} uses unknown tools {unknown}")
        return self


class Registry(BaseModel):
    """The whole agents.yaml file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shared_rules: str = Field(min_length=20)
    agents: list[AgentSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> "Registry":
        dupes = [n for n, c in Counter(a.name for a in self.agents).items() if c > 1]
        if dupes:
            raise ValueError(f"duplicate agent names: {dupes}")
        return self

    def get(self, name: str) -> AgentSpec:
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise KeyError(f"no agent named {name!r} in agents.yaml")


@cache
def load_registry(path: Path = AGENTS_FILE) -> Registry:
    """Parse and validate agents.yaml (cached per path)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Registry.model_validate(data)


if __name__ == "__main__":
    registry = load_registry()
    teams = Counter(a.team for a in registry.agents)
    kinds = Counter(a.kind for a in registry.agents)
    print(f"agents.yaml OK: {len(registry.agents)} agents")
    print("  by team:", dict(teams))
    print("  by kind:", dict(kinds))
