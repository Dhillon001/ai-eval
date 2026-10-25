from pathlib import Path

import pytest

from ai_eval.adapters import build_adapter
from ai_eval.models import EvalCase
from ai_eval.runner import run_suite
from ai_eval.scorers import build_scorer
from ai_eval.scorers.trajectory import TrajectoryScorer
from ai_eval.store import RunStore

ROOT = Path(__file__).resolve().parents[1]


def test_build_scorer_trajectory_with_options():
    scorer = build_scorer({"type": "trajectory", "mode": "subset", "match_args": True, "threshold": 0.5})
    assert isinstance(scorer, TrajectoryScorer)
    assert (scorer.mode, scorer.match_args, scorer.threshold) == ("subset", True, 0.5)
    assert scorer.name == "trajectory:subset"


def test_build_scorer_trajectory_rejects_bad_mode():
    with pytest.raises(ValueError, match="Unknown trajectory mode"):
        build_scorer({"type": "trajectory", "mode": "fuzzy"})


def test_mock_adapter_emits_tool_calls():
    calls = [{"name": "lookup_order", "args": {"order_id": "A1"}}]
    case = EvalCase(id="1", input="q", metadata={"mock_tool_calls": calls})
    run = build_adapter("mock").run(case)
    assert run.trace["tool_calls"] == calls
    assert run.trace["adapter"] == "mock"


def test_mock_adapter_tool_calls_do_not_alias_case_metadata():
    calls = [{"name": "lookup_order", "args": {"order_id": "A1"}}]
    case = EvalCase(id="1", input="q", metadata={"mock_tool_calls": calls})
    run = build_adapter("mock").run(case)
    run.trace["tool_calls"][0]["args"]["order_id"] = "changed"
    assert case.metadata["mock_tool_calls"][0]["args"]["order_id"] == "A1"


def test_mock_adapter_without_tool_calls_leaves_trace_alone():
    run = build_adapter("mock").run(EvalCase(id="1", input="q"))
    assert "tool_calls" not in run.trace


def test_mock_adapter_passes_invalid_tool_calls_through_for_scorer_to_flag():
    case = EvalCase(id="1", input="q", metadata={"mock_tool_calls": "search", "expected_tools": ["search"]})
    run = build_adapter("mock").run(case)
    assert run.trace["tool_calls"] == "search"
    s = TrajectoryScorer().score(case, run)
    assert not s.passed and "invalid tool calls" in s.details["error"]


def test_agent_suite_runs_end_to_end_with_mock(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    _, summary, outcomes = run_suite(
        suite_path=ROOT / "configs" / "agent_suite.yaml",
        project_root=ROOT,
        adapter_name="mock",
        store=RunStore(tmp_path / "test.db"),
    )
    assert summary.total == 6
    assert summary.pass_rate == 1.0
    by_id = {o.case_id: {s.scorer: s for s in o.scores} for o in outcomes}

    exact = by_id["agent-order-status"]
    assert all(s.score == 1.0 for s in exact.values())

    refund = by_id["agent-refund-extra-step"]
    assert refund["trajectory:subset"].score == 1.0
    assert refund["trajectory:any_order"].score == pytest.approx(2 / 3)
    assert refund["trajectory:exact_order"].details["extra"][0]["name"] == "check_refund_policy"

    weather = by_id["agent-weather-parallel"]
    assert weather["trajectory:any_order"].score == 1.0
    assert weather["trajectory:exact_order"].score == 0.5

    assert by_id["agent-kb-search"]["trajectory:subset"].score == 1.0
    assert all(s.score == 1.0 for s in by_id["agent-no-tools"].values())
    assert all("skipped" in s.details for s in by_id["agent-no-expectation"].values())
