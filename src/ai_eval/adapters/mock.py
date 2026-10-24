from __future__ import annotations

from typing import Any

from ai_eval.adapters.base import SystemAdapter
from ai_eval.models import EvalCase, RunResult


class MockAdapter(SystemAdapter):
    """Deterministic adapter for CI and local dev without API keys.

    If a case sets ``metadata["mock_retrieved_ids"]``, the ids are emitted as
    ``trace["retrieved_ids"]`` so retrieval scorers can be exercised offline.
    """

    name = "mock"

    def __init__(self, template: str = "echo:{input}"):
        self.template = template

    def run(self, case: EvalCase) -> RunResult:
        text = self.template.replace("{input}", case.input)
        if case.context:
            text = f"{text}\n[context used]"
        trace: dict[str, Any] = {"adapter": "mock"}
        retrieved = case.metadata.get("mock_retrieved_ids")
        if retrieved is not None:
            trace["retrieved_ids"] = list(retrieved) if isinstance(retrieved, (list, tuple)) else retrieved
        return RunResult(
            case_id=case.id,
            output=text,
            trace=trace,
            latency_ms=1.0,
        )
