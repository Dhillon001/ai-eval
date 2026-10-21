from __future__ import annotations

from abc import ABC, abstractmethod

from ai_eval.models import EvalCase, RunResult, ScoreResult


class Scorer(ABC):
    name: str

    @abstractmethod
    def score(self, case: EvalCase, run: RunResult) -> ScoreResult:
        ...
