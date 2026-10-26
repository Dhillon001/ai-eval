from __future__ import annotations

from typing import Any

from ai_eval.cache import JudgeCache
from ai_eval.scorers.base import Scorer
from ai_eval.scorers.checklist import ChecklistScorer
from ai_eval.scorers.exact import ExactMatchScorer
from ai_eval.scorers.llm_judge import LLMJudgeScorer
from ai_eval.scorers.retrieval import RetrievalScorer
from ai_eval.scorers.trajectory import TrajectoryScorer


def build_scorer(spec: dict[str, Any], cache: JudgeCache | None = None) -> Scorer:
    """Build a scorer from a suite spec. `cache` is handed to LLM judges unless the spec opts out
    with ``cache: false``."""
    kind = spec.get("type") or spec.get("name")
    opts = {k: v for k, v in spec.items() if k not in ("type", "name", "cache")}
    if kind in ("llm_judge", "judge", "faithfulness") and spec.get("cache", True) is not False:
        opts["cache"] = cache
    if kind == "exact":
        return ExactMatchScorer(**opts)
    if kind == "checklist":
        return ChecklistScorer(**opts)
    if kind in ("llm_judge", "judge"):
        return LLMJudgeScorer(**opts)
    if kind == "faithfulness":
        return LLMJudgeScorer(mode="faithfulness", **opts)
    if kind == "retrieval":
        return RetrievalScorer(**opts)
    if kind == "trajectory":
        return TrajectoryScorer(**opts)
    raise ValueError(f"Unknown scorer type: {kind}")
