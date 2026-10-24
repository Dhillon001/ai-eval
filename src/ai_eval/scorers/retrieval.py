from __future__ import annotations

import re
from typing import Any

from ai_eval.models import EvalCase, RunResult, ScoreResult
from ai_eval.scorers.base import Scorer

METRICS = ("recall", "mrr", "hit_rate")

# Pass threshold used when none is configured. Recall and hit rate default to "everything
# expected was found"; MRR defaults to "first relevant doc ranked in the top 2".
DEFAULT_THRESHOLDS = {"recall": 1.0, "hit_rate": 1.0, "mrr": 0.5}

_RECALL_AT = re.compile(r"^recall@(\d+|k)$")


def _parse_metric(metric: str, k: int | None) -> tuple[str, int | None]:
    """Normalize a metric name. Accepts ``recall``, ``recall@k``, ``recall@5`` (sets k),
    ``mrr``, ``hit_rate`` / ``hit_rate@k`` / ``hit@k``."""
    m = metric.strip().lower().replace("-", "_")
    match = _RECALL_AT.match(m)
    if match:
        if match.group(1) != "k":
            k = int(match.group(1))
        return "recall", k
    for prefix, name in (("hit_rate@", "hit_rate"), ("hit@", "hit_rate"), ("mrr@", "mrr")):
        if m.startswith(prefix):
            suffix = m[len(prefix):]
            if suffix != "k":
                if not suffix.isdigit():
                    raise ValueError(f"Invalid retrieval metric: {metric!r}")
                k = int(suffix)
            return name, k
    if m == "hit":
        m = "hit_rate"
    if m not in METRICS:
        raise ValueError(f"Unknown retrieval metric {metric!r}; expected one of {', '.join(METRICS)}")
    return m, k


def _as_id_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"expected a list of ids, got {type(value).__name__}")
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        sid = str(item)
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


class RetrievalScorer(Scorer):
    """Score a retriever by comparing ``run.trace["retrieved_ids"]`` (ranked) with
    ``case.metadata["expected_doc_ids"]`` (the relevant set).

    Metrics (all computed over the top ``k`` retrieved ids; ``k=None`` uses the whole list):

    - ``recall`` (``recall@k``): fraction of expected ids present in the top k.
    - ``hit_rate``: 1.0 if any expected id is in the top k, else 0.0.
    - ``mrr``: reciprocal rank of the first expected id in the top k (0.0 if none).

    Cases without retrieved ids in the trace or without expected ids are skipped (pass).
    """

    name = "retrieval"

    def __init__(
        self,
        metric: str = "recall",
        k: int | None = None,
        threshold: float | None = None,
        trace_key: str = "retrieved_ids",
        expected_key: str = "expected_doc_ids",
    ):
        self.metric, self.k = _parse_metric(metric, k)
        if self.k is not None and (not isinstance(self.k, int) or self.k < 1):
            raise ValueError(f"k must be a positive integer, got {self.k!r}")
        self.threshold = DEFAULT_THRESHOLDS[self.metric] if threshold is None else float(threshold)
        self.trace_key = trace_key
        self.expected_key = expected_key
        label = self.metric if self.k is None else f"{self.metric}@{self.k}"
        self.name = f"retrieval:{label}"

    def _skip(self, reason: str) -> ScoreResult:
        return ScoreResult(scorer=self.name, passed=True, score=1.0, details={"skipped": reason})

    def score(self, case: EvalCase, run: RunResult) -> ScoreResult:
        if run.error:
            return ScoreResult(scorer=self.name, passed=False, score=0.0, details={"error": run.error})

        try:
            expected = _as_id_list(case.metadata.get(self.expected_key))
            retrieved = _as_id_list(run.trace.get(self.trace_key))
        except TypeError as e:
            return ScoreResult(
                scorer=self.name,
                passed=False,
                score=0.0,
                details={"error": f"invalid retrieval ids: {e}"},
            )
        if not expected:
            return self._skip(f"no {self.expected_key} in case metadata")
        if retrieved is None:
            return self._skip(f"no {self.trace_key} in run trace")

        top = retrieved if self.k is None else retrieved[: self.k]
        expected_set = set(expected)
        found = [d for d in top if d in expected_set]
        first_rank = next((i + 1 for i, d in enumerate(top) if d in expected_set), None)

        if self.metric == "recall":
            value = len(found) / len(expected_set)
        elif self.metric == "hit_rate":
            value = 1.0 if found else 0.0
        else:  # mrr
            value = 1.0 / first_rank if first_rank else 0.0

        return ScoreResult(
            scorer=self.name,
            passed=value >= self.threshold,
            score=value,
            details={
                "metric": self.metric,
                "k": self.k,
                "threshold": self.threshold,
                "expected": expected,
                "retrieved": top,
                "found": found,
                "missing": [d for d in expected if d not in set(found)],
                "first_relevant_rank": first_rank,
            },
        )
