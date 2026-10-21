from __future__ import annotations

from typing import Any

from ai_eval.adapters.base import SystemAdapter
from ai_eval.adapters.chat_adapter import AnthropicAdapter, OpenAIAdapter
from ai_eval.adapters.mock import MockAdapter

ADAPTERS = ("mock", "openai", "anthropic")


def build_adapter(name: str, options: dict[str, Any] | None = None) -> SystemAdapter:
    options = dict(options or {})
    if name == "mock":
        return MockAdapter(**{k: v for k, v in options.items() if k in ("template",)})
    options.pop("template", None)
    if name in ("openai", "openai_compat"):
        return OpenAIAdapter(**options)
    if name in ("anthropic", "claude"):
        return AnthropicAdapter(**options)
    raise ValueError(f"Unknown adapter: {name}. Use one of {', '.join(ADAPTERS)}.")
