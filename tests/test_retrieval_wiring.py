from pathlib import Path

import pytest

from ai_eval.adapters import build_adapter
from ai_eval.models import EvalCase
from ai_eval.runner import run_suite
from ai_eval.scorers import build_scorer
from ai_eval.scorers.retrieval import RetrievalScorer
from ai_eval.store import RunStore

ROOT = Path(__file__).resolve().parents[1]


def test_build_scorer_retrieval_with_options():
    scorer = build_scorer({"type": "retrieval", "metric": "mrr", "k": 3, "threshold": 0.25})
    assert isinstance(scorer, RetrievalScorer)
    assert (scorer.metric, scorer.k, scorer.threshold) == ("mrr", 3, 0.25)
    assert scorer.name == "retrieval:mrr@3"


def test_build_scorer_retrieval_rejects_bad_metric():
    with pytest.raises(ValueError, match="Unknown retrieval metric"):
        build_scorer({"type": "retrieval", "metric": "ndcg"})


def test_mock_adapter_emits_retrieved_ids():
    case = EvalCase(id="1", input="q", metadata={"mock_retrieved_ids": ["a", "b"]})
    run = build_adapter("mock").run(case)
    assert run.trace["retrieved_ids"] == ["a", "b"]
    assert run.trace["adapter"] == "mock"


def test_mock_adapter_without_retrieved_ids_leaves_trace_alone():
    run = build_adapter("mock").run(EvalCase(id="1", input="q"))
    assert run.trace == {"adapter": "mock"}


def test_mock_adapter_passes_invalid_ids_through_for_scorer_to_flag():
    case = EvalCase(id="1", input="q", metadata={"mock_retrieved_ids": "abc", "expected_doc_ids": ["a"]})
    run = build_adapter("mock").run(case)
    assert run.trace["retrieved_ids"] == "abc"
    s = RetrievalScorer().score(case, run)
    assert not s.passed and "invalid retrieval ids" in s.details["error"]


def test_retrieval_suite_runs_end_to_end_with_mock(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    _, summary, outcomes = run_suite(
        suite_path=ROOT / "configs" / "retrieval_suite.yaml",
        project_root=ROOT,
        adapter_name="mock",
        store=RunStore(tmp_path / "test.db"),
    )
    assert summary.total == 5
    assert summary.pass_rate == 1.0
    by_id = {o.case_id: {s.scorer: s for s in o.scores} for o in outcomes}

    canada = by_id["ret-shipping-canada"]
    assert canada["retrieval:recall@3"].score == 1.0
    assert canada["retrieval:hit_rate@1"].score == 0.0
    assert canada["retrieval:mrr"].score == 0.5

    warranty = by_id["ret-warranty"]["retrieval:recall@3"]
    assert warranty.score == 1.0 and warranty.details["first_relevant_rank"] == 1

    assert all("skipped" in s.details for s in by_id["ret-no-expectation"].values())
