"""LLM agent nodes. Each node: state -> partial update dict."""

import re
import time
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from devteam import prompts
from devteam.state import DevState

# Switch provider here only: "groq" (primary) or "gemini" (fallback).
PROVIDER = "groq"
MODEL_NAMES = {
    "groq": "openai/gpt-oss-120b",  # GROQ_API_KEY; large free daily quota
    "gemini": "gemini-3.5-flash",  # GOOGLE_API_KEY; free tier ~20 requests/model/day
}
MODEL_NAME = MODEL_NAMES[PROVIDER]
RETRY_WAITS = [10, 20, 40]  # seconds to wait after each retryable error
MAX_OUTPUT_CHARS = 3000  # keep only the tail of test output in prompts (lean context)


@lru_cache(maxsize=1)
def get_llm():
    """Create the chat model once. Switching provider is a change to PROVIDER only."""
    load_dotenv()  # reads GROQ_API_KEY / GOOGLE_API_KEY from .env
    if PROVIDER == "groq":
        return ChatGroq(
            model=MODEL_NAME,
            temperature=0.2,
            reasoning_effort="low",  # gpt-oss reasons before answering; low keeps it fast
            timeout=120,  # seconds per request, so a hung call fails instead of waiting forever
            max_retries=0,  # we do our own backoff in call_llm, so retries don't stack up
        )
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.2,
        thinking_level="low",  # Gemini 3's default (high) is many times slower; low is enough here
        timeout=120,  # seconds per request, so a hung call fails instead of waiting forever
        max_retries=0,  # we do our own backoff in call_llm, so retries don't stack up
    )


def _should_retry(exc: Exception) -> bool:
    """True for temporary 'slow down' errors: 429 quota and 503 model overloaded."""
    text = str(exc)
    return isinstance(exc, ModelRateLimitError) or any(
        code in text for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")
    )


def _is_daily_quota(exc: Exception) -> bool:
    """True if the 429 is the per-day free-tier cap: waiting seconds won't help."""
    return "PerDay" in str(exc) or "per day" in str(exc).lower()


def call_llm(system: str, user: str) -> str:
    """Send one system + user message, retrying on 429/503. Returns the reply text."""
    messages = [SystemMessage(system), HumanMessage(user)]
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            return get_llm().invoke(messages).text
        except Exception as exc:
            if not _should_retry(exc):
                raise
            if _is_daily_quota(exc):
                raise RuntimeError(
                    f"Daily free-tier quota for {MODEL_NAME} is used up; retrying today won't "
                    "help. Switch PROVIDER (or MODEL_NAMES) to another model/provider, "
                    "or wait for the reset."
                ) from exc
            if attempt == len(RETRY_WAITS):
                raise RuntimeError(
                    f"Still rate-limited/overloaded after {len(RETRY_WAITS)} retries "
                    f"(free-tier quota used up, or model busy?). Last error: {exc}"
                ) from exc
            wait = RETRY_WAITS[attempt]
            print(f"  [retry] {type(exc).__name__}: waiting {wait}s ({attempt + 1}/{len(RETRY_WAITS)})")
            time.sleep(wait)
    raise AssertionError("unreachable")


def strip_code_fences(text: str) -> str:
    """Return the code inside ```python ... ``` fences, or the text itself if unfenced."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip() + "\n"


def _preview(text: str, lines: int = 6) -> str:
    head = text.strip().splitlines()[:lines]
    return "\n".join(f"    | {line}" for line in head)


def architect(state: DevState) -> dict:
    """task -> spec (signatures, behaviour, edge cases, errors, examples). No code."""
    print(f"[architect] task={state['task']!r}")
    spec = call_llm(prompts.ARCHITECT_SYSTEM, prompts.ARCHITECT_USER.format(task=state["task"]))
    print(f"[architect] spec: {len(spec.splitlines())} lines\n{_preview(spec)}")
    return {"spec": spec}


def test_writer(state: DevState) -> dict:
    """spec -> test_solution.py source."""
    print("[test_writer] writing tests from spec")
    reply = call_llm(prompts.TEST_WRITER_SYSTEM, prompts.TEST_WRITER_USER.format(spec=state["spec"]))
    tests = strip_code_fences(reply)
    n_tests = len(re.findall(r"^\s*def test_", tests, re.MULTILINE))
    print(f"[test_writer] {n_tests} test functions, {len(tests.splitlines())} lines")
    return {"tests": tests}


def _tail(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Last `limit` characters of text; pytest's summary is at the end."""
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


def coder(state: DevState) -> dict:
    """spec (+ previous code, diagnosis, check output on retries) -> solution.py source."""
    iteration = state.get("iteration", 0) + 1
    is_retry = bool(state.get("code"))
    print(f"[coder] iteration={iteration} retry={is_retry}")

    user = prompts.CODER_USER.format(spec=state["spec"])
    if is_retry:
        user += prompts.CODER_RETRY.format(
            code=state["code"],
            diagnosis=state.get("diagnosis") or "(none)",
            test_output=_tail(state.get("test_output") or "(none)"),
            lint_output=state.get("lint_output") or "(none)",
        )

    code = strip_code_fences(call_llm(prompts.CODER_SYSTEM, user))
    print(f"[coder] solution.py: {len(code.splitlines())} lines\n{_preview(code)}")
    return {"code": code, "iteration": iteration}
