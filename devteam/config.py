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
