from __future__ import annotations

import time
from typing import Any

from ai_eval.adapters.base import SystemAdapter
from ai_eval.llm import LLMError, build_user_prompt, chat, make_client
from ai_eval.models import EvalCase, RunResult


class ChatAdapter(SystemAdapter):
    """Single-turn chat adapter for any supported provider (openai, anthropic)."""

    provider = "openai"
    default_model = "gpt-4o-mini"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        client: Any = None,
    ):
        self.model = model or self.default_model
        self.base_url = base_url
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.provider

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = make_client(self.provider, self.base_url)
        return self._client

    def run(self, case: EvalCase) -> RunResult:
        try:
            client = self._get_client()
        except LLMError as e:
            return RunResult(case_id=case.id, output="", error=str(e))

        user = build_user_prompt(case.input, case.context)
        start = time.perf_counter()
        try:
            resp = chat(
                client,
                self.provider,
                self.model,
                self.system_prompt,
                user,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            return RunResult(
                case_id=case.id,
                output="",
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(e).__name__}: {e}",
            )
        return RunResult(
            case_id=case.id,
            output=resp.text,
            trace={"provider": self.provider, "model": self.model, "system": self.system_prompt, "user": user},
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )


class OpenAIAdapter(ChatAdapter):
    provider = "openai"
    default_model = "gpt-4o-mini"


class AnthropicAdapter(ChatAdapter):
    provider = "anthropic"
    default_model = "claude-haiku-5-5"
