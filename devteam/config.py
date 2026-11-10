"""v2 settings: provider, model tiers and retry policy. Models are named here and nowhere else."""

from pathlib import Path

# Switch provider here only: "groq" (primary) or "gemini" (fallback).
PROVIDER = "groq"

# agents.yaml says `model_tier: strong | fast`; this maps a tier to a real model per provider.
MODEL_TIERS = {
    "groq": {"strong": "openai/gpt-oss-120b", "fast": "openai/gpt-oss-20b"},
    "gemini": {"strong": "gemini-3.5-flash", "fast": "gemini-3.5-flash"},
}

RETRY_WAITS = [10, 20, 40]  # seconds to wait after each 429/503 before giving up
REQUEST_TIMEOUT = 120  # seconds per LLM request

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_FILE = REPO_ROOT / "agents.yaml"
STACKS_FILE = REPO_ROOT / "stacks.yaml"

# Where each run lives: project files in workspace/<run_id>/, logs + events in runs/<run_id>/.
WORKSPACE_ROOT = REPO_ROOT / "workspace"
RUNS_ROOT = REPO_ROOT / "runs"
CHECKPOINT_DB = RUNS_ROOT / "checkpoints.sqlite"  # SQLite checkpointer: resume a run by its id

# Hard limits (autonomous routing can loop forever without them).
RECURSION_LIMIT = 200  # LangGraph super-steps per invoke
MAX_ACTIVATIONS = 60  # agent activations per run
MAX_FIX_LOOPS = 4  # debugger -> fix -> qa rounds per task
MAX_REVIEW_ROUNDS = 2
MAX_BLOCKED_CYCLES = 3  # blocked -> answer cycles per run
MAX_PARALLEL_BUILDERS = 2  # free-tier rate limits
