"""v2 model access: get_llm(tier) is the only place chat models are created."""

import json
import re
import time
from functools import cache
from typing import Literal

from dotenv import load_dotenv
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from devteam.config import MODEL_TIERS, PROVIDER, REQUEST_TIMEOUT, RETRY_WAITS

Tier = Literal["strong", "fast"]


def model_name(tier: Tier) -> str:
    return MODEL_TIERS[PROVIDER][tier]


@cache
def get_llm(tier: Tier):
    """Create one chat model per tier (cached). Library retries are off; invoke() retries."""
    load_dotenv()  # reads GROQ_API_KEY / GOOGLE_API_KEY from .env
    if PROVIDER == "groq":
        return ChatGroq(
            model=model_name(tier),
            temperature=0.2,
            reasoning_effort="low",
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )
    return ChatGoogleGenerativeAI(
        model=model_name(tier),
        temperature=0.2,
        thinking_level="low",
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


def _should_retry(exc: Exception) -> bool:
    """True for temporary 'slow down' errors: 429 quota and 503 model overloaded."""
    text = str(exc)
    return isinstance(exc, ModelRateLimitError) or any(
        code in text for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")
    )


def _is_daily_quota(exc: Exception) -> bool:
    """True if the 429 is a per-day cap: waiting seconds won't help."""
    return "PerDay" in str(exc) or "per day" in str(exc).lower()


def invoke(tier: Tier, system: str, user: str) -> str:
    """Send one system + user message to the tier's model, retrying 429/503. Returns reply text."""
    messages = [SystemMessage(system), HumanMessage(user)]
    return invoke_messages(get_llm(tier), messages, tier).text


def _recover_phantom_tool_call(exc: Exception, tool_names: set[str]) -> AIMessage | None:
    """Groq + GPT-OSS sometimes sends its final JSON answer as a call to a tool that doesn't exist
    (e.g. "json"); Groq rejects it with 400 tool_use_failed. The rejected call's arguments ARE the
    intended answer, so return them as a plain text reply. Calls to real tools are not recovered."""
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(error, dict) or error.get("code") != "tool_use_failed":
        return None
    raw = error.get("failed_generation") or ""
    try:
        generation = json.loads(raw)
        name = generation.get("name") if isinstance(generation, dict) else None
        content = json.dumps(generation.get("arguments", generation)) if name else None
    except json.JSONDecodeError:
        # e.g. two JSON objects inside "arguments": keep the raw text; parse_result finds the result.
        match = re.match(r'\s*\{\s*"name"\s*:\s*"([^"]+)"', raw)
        name, content = (match.group(1), raw) if match else (None, None)
    if not name or name in tool_names:
        return None
    print(f"  [llm] recovered final answer sent as phantom tool {name!r}")
    return AIMessage(content=content)


def invoke_messages(
    model: Runnable, messages: list[BaseMessage], tier: Tier, tool_names: set[str] | None = None
) -> AIMessage:
    """Invoke any chat model (e.g. one with tools bound) on a message list, retrying 429/503.

    `tool_names` are the tools bound to `model`; a call to any other name is treated as text.
    """
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            return model.invoke(messages)
        except Exception as exc:
            recovered = _recover_phantom_tool_call(exc, tool_names or set())
            if recovered is not None:
                return recovered
            if not _should_retry(exc):
                raise
            if _is_daily_quota(exc):
                raise RuntimeError(
                    f"Daily quota for {model_name(tier)} is used up; switch PROVIDER or "
                    "MODEL_TIERS in devteam/config.py, or wait for the reset."
                ) from exc
            if attempt == len(RETRY_WAITS):
                raise RuntimeError(
                    f"Still rate-limited/overloaded after {len(RETRY_WAITS)} retries. "
                    f"Last error: {exc}"
                ) from exc
            wait = RETRY_WAITS[attempt]
            print(f"  [retry] {type(exc).__name__}: waiting {wait}s ({attempt + 1}/{len(RETRY_WAITS)})")
            time.sleep(wait)
    raise AssertionError("unreachable")
