from __future__ import annotations

from abc import ABC, abstractmethod

from ai_eval.models import EvalCase, RunResult


class SystemAdapter(ABC):
    name: str

    @abstractmethod
    def run(self, case: EvalCase) -> RunResult:
        ...
