from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_eval.adapters.mock import MockAdapter
from ai_eval.calibrate import calibrate_suite
from ai_eval.cli import app
from ai_eval.dataset import apply_labels, load_cases, load_labels, load_suite
from ai_eval.models import EvalCase, RunResult
from ai_eval.scorers.llm_judge import LLMJudgeScorer

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/calibration_suite.yaml"
CASES = ROOT / "datasets/examples/calibration_v1.jsonl"
LABELS = ROOT / "datasets/examples/calibration_v1.labels.jsonl"


def test_calibration_set_is_complete_and_offline():
    cases = load_cases(CASES)
    labels = load_labels(LABELS)
    assert len(cases) == 20
    assert len({c.id for c in cases}) == 20
    assert set(labels) == {c.id for c in cases}, "every case needs exactly one human label"
    for case in apply_labels(cases, labels):
        assert "calibration" in case.metadata["tags"], case.id
        assert case.metadata.get("judge_criteria"), case.id
        assert isinstance(case.metadata.get("mock_output"), str) and case.metadata["mock_output"], case.id
        assert isinstance(case.metadata.get("mock_judge"), dict), case.id
        assert case.human_label is not None and case.human_label.score is not None, case.id
    # Both classes present on both sides, or kappa is undefined.
    human = [labels[c.id].passed for c in cases]
    judge = [c.metadata["mock_judge"]["pass"] for c in cases]
    assert 0 < sum(human) < len(human) and 0 < sum(judge) < len(judge)


def test_calibration_suite_runs_offline_with_known_metrics():
    res = calibrate_suite(SUITE, project_root=ROOT, labels_path=LABELS, use_cache=False)
    assert res.labeled == res.n == 20 and not res.errors
    assert (res.confusion.tp, res.confusion.fp, res.confusion.fn, res.confusion.tn) == (11, 3, 1, 5)
    assert res.accuracy == pytest.approx(0.8)
    assert res.kappa == pytest.approx(0.26 / 0.46)
    assert res.pearson is not None and res.pearson > 0.5
    assert {i.case_id for i in res.disagreements} == {
        "exchange-size", "ship-express", "international-returns", "warranty-water",
    }
    assert res.judge.startswith("mock:")


def test_calibrate_cli_on_example_suite(monkeypatch):
    monkeypatch.chdir(ROOT)
    runner = CliRunner()
    args = ["calibrate", "--suite", str(SUITE), "--labels", str(LABELS), "--no-cache"]
    assert runner.invoke(app, [*args, "--min-kappa", "0.5"]).exit_code == 0
    assert runner.invoke(app, [*args, "--min-kappa", "0.7"]).exit_code == 1


def test_calibration_suite_uses_mock_adapter_and_judge():
    suite = load_suite(SUITE, project_root=ROOT)
    assert suite.adapter["type"] == "mock"
    assert suite.scorers[0]["type"] == "llm_judge" and suite.scorers[0]["provider"] == "mock"


def test_mock_adapter_replays_stored_output():
    adapter = MockAdapter(template="T:{input}")
    stored = EvalCase(id="a", input="q", context="ctx", metadata={"mock_output": "stored answer"})
    assert adapter.run(stored).output == "stored answer"
    assert adapter.run(EvalCase(id="b", input="q")).output == "T:q"


def _run(output: str) -> RunResult:
    return RunResult(case_id="c", output=output)


def test_mock_judge_replays_stored_verdict():
    judge = LLMJudgeScorer(provider="mock")
    case = EvalCase(id="c", input="q", metadata={"mock_judge": {"pass": False, "score": 0.9, "reason": "r"}})
    res = judge.score(case, _run("anything"))
    assert res.passed is False and res.score == pytest.approx(0.9)
    assert res.details["parsed"]["reason"] == "r" and res.details["judge"] == "mock:replay"

    # Without "pass", the threshold decides.
    case = EvalCase(id="c", input="q", metadata={"mock_judge": {"score": 0.6}})
    assert LLMJudgeScorer(provider="mock", pass_threshold=0.5).score(case, _run("x")).passed is True
    assert LLMJudgeScorer(provider="mock", pass_threshold=0.7).score(case, _run("x")).passed is False


def test_mock_judge_falls_back_to_checks_and_errors_without_either():
    judge = LLMJudgeScorer(provider="mock")
    case = EvalCase(id="c", input="q", checks=["30 days", "refund"])
    full = judge.score(case, _run("Full REFUND within 30 days."))
    assert full.passed and full.score == 1.0
    half = judge.score(case, _run("within 30 days"))
    assert not half.passed and half.score == pytest.approx(0.5)

    bare = judge.score(EvalCase(id="c", input="q"), _run("x"))
    assert not bare.passed and "mock judge needs" in bare.details["error"]


def test_mock_judge_never_builds_a_client(monkeypatch):
    import ai_eval.scorers.llm_judge as mod

    def boom(*_a, **_k):
        raise AssertionError("mock judge must not call a provider")

    monkeypatch.setattr(mod, "make_client", boom)
    monkeypatch.setattr(mod, "chat", boom)
    case = EvalCase(id="c", input="q", metadata={"mock_judge": {"pass": True, "score": 1.0}})
    assert LLMJudgeScorer(provider="mock").score(case, _run("x")).passed
