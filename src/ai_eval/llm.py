"""Thin provider-agnostic chat helper shared by adapters and the LLM judge."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

PROVIDERS = ("openai", "anthropic")

# Retry policy for provider calls. The SDKs' own retries are disabled in make_client so
# attempts are not multiplied; these defaults roughly match what the SDKs would do.
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 0.5  # seconds; doubled each attempt
DEFAULT_RETRY_MAX_DELAY = 20.0  # cap for a single wait (also caps Retry-After)
RETRYABLE_STATUS = frozenset({408, 429})


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retries: int = 0


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def is_retryable(exc: BaseException) -> bool:
    """True for transient provider failures: HTTP 408/429/5xx, timeouts, and connection errors.

    Detection is duck-typed (status codes and exception class names) so it works for the
    openai and anthropic SDKs and httpx without importing any of them.
    """
    if isinstance(exc, LLMError):
        return False
    code = _status_code(exc)
    if code is not None:
        return code in RETRYABLE_STATUS or 500 <= code < 600
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    names = {cls.__name__ for cls in type(exc).__mro__}
    return any("Timeout" in n for n in names) or bool(names & {"APIConnectionError", "ConnectError"})


def _retry_after(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
        return max(0.0, float(value)) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def backoff_delay(attempt: int, base: float, cap: float, rng: Callable[[], float] = random.random) -> float:
    """Exponential backoff with jitter: a random wait in [50%, 100%] of min(cap, base * 2**attempt)."""
    ceiling = min(cap, base * (2**attempt))
    return ceiling * (0.5 + 0.5 * rng())


def call_with_retries(
    fn: Callable[[], Any],
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, int]:
    """Call `fn`, retrying transient failures (see `is_retryable`). Returns (result, retries_used).

    Non-retryable errors, and the last error once `max_retries` is exhausted, are re-raised.
    A server-sent ``Retry-After`` header is honoured (capped at `max_delay`).
    """
    attempt = 0
    while True:
        try:
            return fn(), attempt
        except Exception as exc:
            if attempt >= max_retries or not is_retryable(exc):
                raise
            hinted = _retry_after(exc)
            delay = min(max_delay, hinted) if hinted is not None else backoff_delay(attempt, base_delay, max_delay)
            sleep(delay)
            attempt += 1


def make_client(provider: str, base_url: str | None = None) -> Any:
    """Build an SDK client for `provider`, raising LLMError with a fix-it hint on failure."""
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError("Install openai: pip install -e '.[openai]'") from e
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set")
        kwargs: dict[str, Any] = {"api_key": key, "max_retries": 0}  # retries handled in chat()
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)
    if provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise LLMError("Install anthropic: pip install -e '.[anthropic]'") from e
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        kwargs = {"api_key": key, "max_retries": 0}  # retries handled in chat()
        if base_url:
            kwargs["base_url"] = base_url
        return Anthropic(**kwargs)
    raise LLMError(f"Unknown provider: {provider}. Use one of {PROVIDERS}.")


def chat(
    client: Any,
    provider: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    sleep: Callable[[float], None] = time.sleep,
) -> ChatResponse:
    """One chat completion. Transient failures (429/5xx/timeouts) are retried up to `max_retries`
    times with exponential backoff and jitter; `max_retries=0` disables retrying."""
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider: {provider}. Use one of {PROVIDERS}.")

    def retrying(create: Callable[..., Any], **kwargs: Any) -> tuple[Any, int]:
        return call_with_retries(
            lambda: create(**kwargs), max_retries=max_retries, base_delay=retry_base_delay, sleep=sleep
        )

    if provider == "openai":
        resp, retries = retrying(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        usage = getattr(resp, "usage", None)
        return ChatResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            retries=retries,
        )
    if provider == "anthropic":
        resp, retries = retrying(
            client.messages.create,
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "text") == "text")
        usage = getattr(resp, "usage", None)
        return ChatResponse(
            text=text,
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            retries=retries,
        )
    raise LLMError(f"Unknown provider: {provider}. Use one of {PROVIDERS}.")


def build_user_prompt(input_text: str, context: str | None) -> str:
    if context:
        return f"Context:\n{context}\n\nQuestion:\n{input_text}"
    return input_text
