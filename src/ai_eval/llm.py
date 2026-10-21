"""Thin provider-agnostic chat helper shared by adapters and the LLM judge."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

PROVIDERS = ("openai", "anthropic")


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


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
        kwargs: dict[str, Any] = {"api_key": key}
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
        kwargs = {"api_key": key}
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
) -> ChatResponse:
    if provider == "openai":
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        usage = getattr(resp, "usage", None)
        return ChatResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
    if provider == "anthropic":
        resp = client.messages.create(
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
        )
    raise LLMError(f"Unknown provider: {provider}. Use one of {PROVIDERS}.")


def build_user_prompt(input_text: str, context: str | None) -> str:
    if context:
        return f"Context:\n{context}\n\nQuestion:\n{input_text}"
    return input_text
