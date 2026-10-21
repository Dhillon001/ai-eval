from __future__ import annotations

from ai_eval.adapters.base import SystemAdapter
from ai_eval.models import EvalCase, RunResult


class MockAdapter(SystemAdapter):
    """Deterministic adapter for CI and local dev without API keys."""

    name = "mock"

    def __init__(self, template: str = "echo:{input}"):
        self.template = template

    def run(self, case: EvalCase) -> RunResult:
        text = self.template.replace("{input}", case.input)
        if case.context:
            text = f"{text}\n[context used]"
        return RunResult(
            case_id=case.id,
            output=text,
            trace={"adapter": "mock"},
            latency_ms=1.0,
        )
