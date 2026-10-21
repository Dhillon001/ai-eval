from __future__ import annotations

import os
import time
from typing import Any

from ai_eval.adapters.base import SystemAdapter
from ai_eval.models import EvalCase, RunResult


class OpenAICompatAdapter(SystemAdapter):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ):
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.temperature = temperature

    def run(self, case: EvalCase) -> RunResult:
        try:
            from openai import OpenAI
        except ImportError as e:
            return RunResult(
                case_id=case.id,
                output="",
                error="Install openai: pip install -e '.[openai]'",
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return RunResult(
                case_id=case.id,
                output="",
                error="OPENAI_API_KEY is not set",
            )

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)

        user_content = case.input
        if case.context:
            user_content = f"Context:\n{case.context}\n\nQuestion:\n{case.input}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return RunResult(
                case_id=case.id,
                output="",
                latency_ms=elapsed,
                error=str(e),
            )

        elapsed = (time.perf_counter() - start) * 1000
        choice = resp.choices[0].message.content or ""
        usage = resp.usage
        return RunResult(
            case_id=case.id,
            output=choice,
            trace={"model": self.model, "messages": messages},
            latency_ms=elapsed,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
