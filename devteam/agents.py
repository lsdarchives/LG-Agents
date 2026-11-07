"""LLM agent nodes. Each node: state -> partial update dict."""

import re
import time
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from devteam import prompts
from devteam.state import DevState

# Change the model here only. If a model is slow or overloaded, try another Flash model.
MODEL_NAME = "gemini-3.5-flash"
RETRY_WAITS = [10, 20, 40]  # seconds to wait after each retryable error


@lru_cache(maxsize=1)
def get_llm():
    """Create the chat model once. Switching provider is a change here only."""
    load_dotenv()  # reads GOOGLE_API_KEY from .env
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


def call_llm(system: str, user: str) -> str:
    """Send one system + user message, retrying on 429/503. Returns the reply text."""
    messages = [SystemMessage(system), HumanMessage(user)]
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            return get_llm().invoke(messages).text
        except Exception as exc:
            if not _should_retry(exc):
                raise
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
