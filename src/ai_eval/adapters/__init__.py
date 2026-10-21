from __future__ import annotations

from typing import Any

from ai_eval.adapters.base import SystemAdapter
from ai_eval.adapters.mock import MockAdapter
from ai_eval.adapters.openai_compat import OpenAICompatAdapter


def build_adapter(name: str, options: dict[str, Any] | None = None) -> SystemAdapter:
    options = options or {}
    if name == "mock":
        return MockAdapter(**{k: v for k, v in options.items() if k in ("template",)})
    if name in ("openai", "openai_compat"):
        return OpenAICompatAdapter(**options)
    raise ValueError(f"Unknown adapter: {name}. Use mock or openai.")
