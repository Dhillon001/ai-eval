from __future__ import annotations

import copy
from typing import Any

from ai_eval.adapters.base import SystemAdapter
from ai_eval.models import EvalCase, RunResult


class MockAdapter(SystemAdapter):
    """Deterministic adapter for CI and local dev without API keys.

    If a case sets ``metadata["mock_retrieved_ids"]``, the ids are emitted as
    ``trace["retrieved_ids"]`` so retrieval scorers can be exercised offline.
    Likewise ``metadata["mock_tool_calls"]`` (a list of ``{name, args}``) is emitted
    as ``trace["tool_calls"]`` for the trajectory scorer.
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
        tool_calls = case.metadata.get("mock_tool_calls")
        if tool_calls is not None:
            # Deep copy so scorers/stores can't mutate the case's metadata via the trace.
            trace["tool_calls"] = copy.deepcopy(list(tool_calls) if isinstance(tool_calls, tuple) else tool_calls)
        return RunResult(
            case_id=case.id,
            output=text,
            trace=trace,
            latency_ms=1.0,
        )
