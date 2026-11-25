from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import ai_eval.calibrate as calibrate_mod
from ai_eval.calibrate import (
    CalibrationItem,
    calibrate_suite,
    cohens_kappa,
    confusion_matrix,
    pearson,
    spearman,
    summarize,
)
from ai_eval.cli import app
from ai_eval.scorers.llm_judge import LLMJudgeScorer
from fakes import FakeOpenAIByPrompt


def _verdict_judge(prompt: str) -> str:
    """Passes outputs that mention GOOD; scores come from a `score=0.x` marker if present."""
    m = re.search(r"score=([0-9.]+)", prompt)
    score = float(m.group(1)) if m else (0.9 if "GOOD" in prompt else 0.1)
    return json.dumps({"pass": "GOOD" in prompt, "score": score, "reason": "fake"})


def _write_suite(tmp_path: Path, rows: list[dict], labels: list[dict] | None = None) -> Path:
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    if labels is not None:
        (tmp_path / "labels.jsonl").write_text(
            "\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8"
        )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: calib\ndataset: cases.jsonl\nadapter:\n  type: mock\n  template: '{input}'\n"
        "scorers:\n  - type: checklist\n  - type: llm_judge\n",
        encoding="utf-8",
    )
    return suite


def _judge() -> LLMJudgeScorer:
    return LLMJudgeScorer(client=FakeOpenAIByPrompt(_verdict_judge))


def test_confusion_and_kappa_basic():
    # judge, human
    cm = confusion_matrix([(True, True), (True, False), (False, True), (False, False), (True, True)])
    assert (cm.tp, cm.fp, cm.fn, cm.tn) == (2, 1, 1, 1)
    # p_o = 0.6, p_e = 0.6*0.6 + 0.4*0.4 = 0.52 -> kappa = 0.08/0.48
    assert cohens_kappa(cm) == pytest.approx(0.08 / 0.48)
    assert cohens_kappa(confusion_matrix([(True, True), (False, False)])) == pytest.approx(1.0)
    assert cohens_kappa(confusion_matrix([(True, True), (True, True)])) is None  # one class only
    assert cohens_kappa(confusion_matrix([])) is None


def test_correlations():
    assert pearson([0.1, 0.5, 0.9], [0.2, 0.6, 1.0]) == pytest.approx(1.0)
    assert pearson([0.1, 0.5, 0.9], [1.0, 0.6, 0.2]) == pytest.approx(-1.0)
    assert pearson([0.5, 0.5], [0.1, 0.9]) is None  # constant series
    assert pearson([0.5], [0.5]) is None
    # monotone but non-linear: Spearman 1, Pearson below 1
    xs, ys = [0.1, 0.2, 0.3, 0.9], [0.1, 0.2, 0.3, 0.31]
    assert spearman(xs, ys) == pytest.approx(1.0)
    assert pearson(xs, ys) < 1.0


def test_summarize_excludes_errors():
    items = [
        CalibrationItem(case_id="a", human_passed=True, judge_passed=True, judge_score=0.9, human_score=1.0),
        CalibrationItem(case_id="b", human_passed=False, judge_passed=True, judge_score=0.8, human_score=0.2),
        CalibrationItem(case_id="c", human_passed=False, error="judge error: boom"),
    ]
    res = summarize(items, suite_name="s", judge="j", unlabeled=4)
    assert (res.labeled, res.n, res.unlabeled) == (3, 2, 4)
    assert res.accuracy == 0.5
    assert [i.case_id for i in res.disagreements] == ["b"]
    assert [i.case_id for i in res.errors] == ["c"]
    assert res.score_pairs == 2


def test_calibrate_suite_with_labels_file(tmp_path):
    rows = [
        {"id": "1", "input": "GOOD answer score=0.9"},
        {"id": "2", "input": "GOOD answer score=0.7"},
        {"id": "3", "input": "bad answer score=0.2"},
        {"id": "4", "input": "bad answer score=0.1", "human_label": {"pass": True}},  # judge disagrees
        {"id": "5", "input": "unlabeled"},
    ]
    labels = [
        {"id": "1", "passed": True, "score": 1.0},
        {"id": "2", "passed": True, "score": 0.8},
        {"id": "3", "passed": False, "score": 0.0},
    ]
    suite = _write_suite(tmp_path, rows, labels)
    res = calibrate_suite(suite, tmp_path, labels_path=tmp_path / "labels.jsonl", judge=_judge())
    assert (res.labeled, res.unlabeled, res.n) == (4, 1, 4)
    assert (res.confusion.tp, res.confusion.fp, res.confusion.fn, res.confusion.tn) == (2, 0, 1, 1)
    assert res.accuracy == 0.75
    assert res.kappa == pytest.approx(0.5)
    assert [i.case_id for i in res.disagreements] == ["4"]
    assert res.score_pairs == 3 and res.pearson == pytest.approx(pearson([0.9, 0.7, 0.2], [1.0, 0.8, 0.0]))
    assert res.items[0].reason == "fake" and res.items[0].output.startswith("GOOD")


def test_calibrate_records_judge_errors(tmp_path):
    class Boom:
        name = "llm_judge"

        def score(self, case, run):
            raise RuntimeError("down")

    suite = _write_suite(tmp_path, [{"id": "1", "input": "x", "human_label": {"pass": True}}])
    res = calibrate_suite(suite, tmp_path, judge=Boom())  # type: ignore[arg-type]
    assert res.n == 0 and res.accuracy is None and res.kappa is None
    assert "judge crashed" in (res.errors[0].error or "")


def test_calibrate_needs_labels_and_a_judge(tmp_path):
    suite = _write_suite(tmp_path, [{"id": "1", "input": "x"}])
    with pytest.raises(ValueError, match="No human-labeled cases"):
        calibrate_suite(suite, tmp_path, judge=_judge())
    suite.write_text("name: c\ndataset: cases.jsonl\nscorers:\n  - type: checklist\n", encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text(json.dumps({"id": "1", "input": "x", "human_label": {"pass": True}}))
    with pytest.raises(ValueError, match="no LLM judge"):
        calibrate_suite(suite, tmp_path)
    with pytest.raises(ValueError, match="not an LLM judge"):
        calibrate_suite(suite, tmp_path, judge_name="checklist")


def test_calibrate_cli(tmp_path, monkeypatch):
    rows = [
        {"id": "1", "input": "GOOD", "human_label": {"pass": True}},
        {"id": "2", "input": "bad", "human_label": {"pass": False}},
        {"id": "3", "input": "bad", "human_label": {"pass": True}},
    ]
    _write_suite(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(calibrate_mod, "build_scorer", lambda spec, cache=None: _judge())
    runner = CliRunner()

    res = runner.invoke(app, ["calibrate", "--suite", "suite.yaml", "--out", "calib.json", "--no-cache"])
    assert res.exit_code == 0, res.output
    assert "Cohen's kappa" in res.output and "disagree" in res.output
    data = json.loads((tmp_path / "calib.json").read_text())
    assert data["accuracy"] == pytest.approx(2 / 3) and data["disagreements"] == ["3"]

    res = runner.invoke(app, ["calibrate", "--suite", "suite.yaml", "--min-kappa", "0.9", "--no-cache"])
    assert res.exit_code == 1
