from __future__ import annotations

from typing import Any

from ai_eval.scorers.base import Scorer
from ai_eval.scorers.checklist import ChecklistScorer
from ai_eval.scorers.exact import ExactMatchScorer
from ai_eval.scorers.llm_judge import LLMJudgeScorer


def build_scorer(spec: dict[str, Any]) -> Scorer:
    kind = spec.get("type") or spec.get("name")
    opts = {k: v for k, v in spec.items() if k not in ("type", "name")}
    if kind == "exact":
        return ExactMatchScorer(**opts)
    if kind == "checklist":
        return ChecklistScorer(**opts)
    if kind in ("llm_judge", "judge"):
        return LLMJudgeScorer(**opts)
    if kind == "faithfulness":
        return LLMJudgeScorer(mode="faithfulness", **opts)
    raise ValueError(f"Unknown scorer type: {kind}")
