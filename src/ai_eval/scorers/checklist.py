from __future__ import annotations

from ai_eval.models import EvalCase, RunResult, ScoreResult
from ai_eval.scorers.base import Scorer


class ChecklistScorer(Scorer):
    """Pass if all required substrings appear in output (case-insensitive)."""

    name = "checklist"

    def __init__(self, case_insensitive: bool = True):
        self.case_insensitive = case_insensitive

    def score(self, case: EvalCase, run: RunResult) -> ScoreResult:
        if run.error:
            return ScoreResult(
                scorer=self.name,
                passed=False,
                score=0.0,
                details={"error": run.error},
            )
        checks = case.checks or []
        if not checks:
            meta_checks = case.metadata.get("checks")
            if isinstance(meta_checks, list):
                checks = [str(c) for c in meta_checks]
        if not checks:
            return ScoreResult(
                scorer=self.name,
                passed=True,
                score=1.0,
                details={"skipped": "no checks"},
            )

        haystack = run.output.lower() if self.case_insensitive else run.output
        missing: list[str] = []
        for item in checks:
            needle = item.lower() if self.case_insensitive else item
            if needle not in haystack:
                missing.append(item)

        passed = len(missing) == 0
        score = (len(checks) - len(missing)) / len(checks)
        return ScoreResult(
            scorer=self.name,
            passed=passed,
            score=score,
            details={"checks": checks, "missing": missing},
        )
