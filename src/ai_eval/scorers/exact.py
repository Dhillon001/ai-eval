from __future__ import annotations

from ai_eval.models import EvalCase, RunResult, ScoreResult
from ai_eval.scorers.base import Scorer


class ExactMatchScorer(Scorer):
    name = "exact"

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def _norm(self, s: str) -> str:
        if not self.normalize:
            return s
        return " ".join(s.strip().lower().split())

    def score(self, case: EvalCase, run: RunResult) -> ScoreResult:
        if run.error:
            return ScoreResult(
                scorer=self.name,
                passed=False,
                score=0.0,
                details={"error": run.error},
            )
        expected = case.expected
        if expected is None:
            return ScoreResult(
                scorer=self.name,
                passed=True,
                score=1.0,
                details={"skipped": "no expected field"},
            )
        ok = self._norm(run.output) == self._norm(expected)
        return ScoreResult(
            scorer=self.name,
            passed=ok,
            score=1.0 if ok else 0.0,
            details={"expected": expected, "got": run.output},
        )
