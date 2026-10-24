from __future__ import annotations

import pytest

from ai_eval.models import EvalCase, RunResult
from ai_eval.scorers.retrieval import RetrievalScorer


def _case(expected=None, **metadata) -> EvalCase:
    if expected is not None:
        metadata["expected_doc_ids"] = expected
    return EvalCase(id="c1", input="q", metadata=metadata)


def _run(retrieved=None, error: str | None = None, **trace) -> RunResult:
    if retrieved is not None:
        trace["retrieved_ids"] = retrieved
    return RunResult(case_id="c1", output="a", trace=trace, error=error)


# --- recall ---------------------------------------------------------------


def test_perfect_recall():
    s = RetrievalScorer(metric="recall").score(_case(["d1", "d2"]), _run(["d2", "x", "d1"]))
    assert s.passed
    assert s.score == 1.0
    assert s.details["missing"] == []
    assert s.scorer == "retrieval:recall"


def test_partial_recall_fails_default_threshold():
    s = RetrievalScorer(metric="recall").score(_case(["d1", "d2"]), _run(["d1", "x", "y"]))
    assert not s.passed
    assert s.score == pytest.approx(0.5)
    assert s.details["found"] == ["d1"]
    assert s.details["missing"] == ["d2"]


def test_partial_recall_passes_custom_threshold():
    scorer = RetrievalScorer(metric="recall", threshold=0.5)
    s = scorer.score(_case(["d1", "d2"]), _run(["d1", "x"]))
    assert s.passed
    assert s.details["threshold"] == 0.5


def test_zero_recall():
    s = RetrievalScorer(metric="recall").score(_case(["d1", "d2"]), _run(["x", "y", "z"]))
    assert not s.passed
    assert s.score == 0.0
    assert s.details["first_relevant_rank"] is None


def test_recall_at_k_truncates_ranked_list():
    scorer = RetrievalScorer(metric="recall@2")
    assert scorer.k == 2
    assert scorer.name == "retrieval:recall@2"
    s = scorer.score(_case(["d1", "d2"]), _run(["d1", "x", "d2"]))
    assert s.score == pytest.approx(0.5)
    assert s.details["retrieved"] == ["d1", "x"]


def test_recall_at_k_placeholder_uses_k_argument():
    scorer = RetrievalScorer(metric="recall@k", k=3)
    assert scorer.k == 3
    s = scorer.score(_case(["d1", "d2"]), _run(["d1", "x", "d2", "y"]))
    assert s.score == 1.0


def test_k_larger_than_list():
    scorer = RetrievalScorer(metric="recall", k=10)
    s = scorer.score(_case(["d1", "d2"]), _run(["d2", "d1"]))
    assert s.passed
    assert s.score == 1.0
    assert s.details["retrieved"] == ["d2", "d1"]


def test_duplicate_retrieved_ids_counted_once():
    s = RetrievalScorer(metric="recall").score(_case(["d1", "d2"]), _run(["d1", "d1", "d1"]))
    assert s.score == pytest.approx(0.5)
    assert s.details["retrieved"] == ["d1"]


def test_ids_compared_as_strings():
    s = RetrievalScorer(metric="recall").score(_case([1, 2]), _run(["2", 1]))
    assert s.score == 1.0


# --- MRR ------------------------------------------------------------------


@pytest.mark.parametrize(
    "retrieved, expected_score, rank",
    [
        (["d1", "x", "y"], 1.0, 1),
        (["x", "d1", "y"], 0.5, 2),
        (["x", "y", "d1"], 1 / 3, 3),
        (["x", "y", "z"], 0.0, None),
    ],
)
def test_mrr_ranking(retrieved, expected_score, rank):
    s = RetrievalScorer(metric="mrr").score(_case(["d1"]), _run(retrieved))
    assert s.score == pytest.approx(expected_score)
    assert s.details["first_relevant_rank"] == rank
    # Default MRR threshold is 0.5: rank 1 or 2 passes.
    assert s.passed == (rank is not None and rank <= 2)


def test_mrr_uses_first_of_several_relevant_docs():
    s = RetrievalScorer(metric="mrr").score(_case(["d1", "d2"]), _run(["x", "x2", "d2", "d1"]))
    assert s.score == pytest.approx(1 / 3)


def test_mrr_at_k_ignores_hits_past_k():
    s = RetrievalScorer(metric="mrr@2").score(_case(["d1"]), _run(["x", "y", "d1"]))
    assert s.score == 0.0
    assert not s.passed


# --- hit rate -------------------------------------------------------------


def test_hit_rate_any_match():
    s = RetrievalScorer(metric="hit_rate").score(_case(["d1", "d2"]), _run(["x", "d2"]))
    assert s.passed
    assert s.score == 1.0


def test_hit_rate_miss():
    s = RetrievalScorer(metric="hit@1").score(_case(["d1"]), _run(["x", "d1"]))
    assert s.score == 0.0
    assert not s.passed


# --- skips and errors -----------------------------------------------------


def test_missing_trace_is_skipped():
    s = RetrievalScorer().score(_case(["d1"]), _run())
    assert s.passed
    assert "skipped" in s.details
    assert "retrieved_ids" in s.details["skipped"]


def test_missing_expected_ids_is_skipped():
    s = RetrievalScorer().score(_case(), _run(["d1"]))
    assert s.passed
    assert "expected_doc_ids" in s.details["skipped"]


def test_empty_expected_ids_is_skipped():
    s = RetrievalScorer().score(_case([]), _run(["d1"]))
    assert s.passed
    assert "skipped" in s.details


def test_empty_retrieved_list_scores_zero():
    s = RetrievalScorer().score(_case(["d1"]), _run([]))
    assert not s.passed
    assert s.score == 0.0


def test_run_error_fails():
    s = RetrievalScorer().score(_case(["d1"]), _run(["d1"], error="boom"))
    assert not s.passed
    assert s.details["error"] == "boom"


def test_non_list_ids_fail_cleanly():
    s = RetrievalScorer().score(_case(["d1"]), _run("d1"))
    assert not s.passed
    assert "invalid retrieval ids" in s.details["error"]


def test_custom_keys():
    scorer = RetrievalScorer(trace_key="docs", expected_key="gold")
    case = EvalCase(id="c1", input="q", metadata={"gold": ["a"]})
    run = RunResult(case_id="c1", output="", trace={"docs": ["a"]})
    assert scorer.score(case, run).score == 1.0


# --- configuration --------------------------------------------------------


@pytest.mark.parametrize(
    "metric, name, k",
    [
        ("recall", "recall", None),
        ("Recall@5", "recall", 5),
        ("hit", "hit_rate", None),
        ("hit-rate@3", "hit_rate", 3),
        ("mrr@10", "mrr", 10),
    ],
)
def test_metric_parsing(metric, name, k):
    scorer = RetrievalScorer(metric=metric)
    assert scorer.metric == name
    assert scorer.k == k


@pytest.mark.parametrize("metric", ["precision", "hit@abc"])
def test_unknown_metric_raises(metric):
    with pytest.raises(ValueError):
        RetrievalScorer(metric=metric)


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_raises(k):
    with pytest.raises(ValueError):
        RetrievalScorer(k=k)
