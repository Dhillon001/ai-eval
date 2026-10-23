from __future__ import annotations

from typing import Any

from ai_eval.adapters.base import SystemAdapter
from ai_eval.adapters.chat_adapter import AnthropicAdapter, OpenAIAdapter
from ai_eval.adapters.http import HTTPAdapter
from ai_eval.adapters.mock import MockAdapter

ADAPTERS = ("mock", "openai", "anthropic", "http")

_HTTP_OPTIONS = ("url", "body_template", "response_path", "headers", "timeout", "method", "trace_path")


def build_adapter(name: str, options: dict[str, Any] | None = None) -> SystemAdapter:
    options = dict(options or {})
    if name == "mock":
        return MockAdapter(**{k: v for k, v in options.items() if k in ("template",)})
    options.pop("template", None)
    if name in ("openai", "openai_compat"):
        return OpenAIAdapter(**options)
    if name in ("anthropic", "claude"):
        return AnthropicAdapter(**options)
    if name == "http":
        unknown = sorted(set(options) - set(_HTTP_OPTIONS))
        if unknown:
            raise ValueError(
                f"Unknown option(s) for http adapter: {', '.join(unknown)}. "
                f"Allowed: {', '.join(_HTTP_OPTIONS)}."
            )
        return HTTPAdapter(**options)
    raise ValueError(f"Unknown adapter: {name}. Use one of {', '.join(ADAPTERS)}.")
