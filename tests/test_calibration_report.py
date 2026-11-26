from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import ai_eval.calibrate as calibrate_mod
from ai_eval.calibrate import CalibrationItem, summarize
from ai_eval.cli import app
from ai_eval.models import EvalCase
from ai_eval.report import kappa_band, write_calibration_report
from ai_eval.scorers.llm_judge import LLMJudgeScorer
from fakes import FakeOpenAIByPrompt

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/calibration_suite.yaml"
LABELS = ROOT / "datasets/examples/calibration_v1.labels.jsonl"


# --- HTML report ---------------------------------------------------------------------------


def _card(doc: str, label: str) -> tuple[str, str]:
    """(value, sub) of the summary card with this label."""
    m = re.search(
        rf"<div class='label'>{re.escape(label)}</div><div class='value'>(.*?)</div><div class='sub'>(.*?)</div>",
        doc,
    )
    assert m, f"no card {label!r}"
    return m.group(1), m.group(2)


def test_calibration_report_renders_summary_and_disagreements(tmp_path):
    items = [
        CalibrationItem(case_id="ok", human_passed=True, judge_passed=True, judge_score=0.9),
        CalibrationItem(case_id="false-pass", human_passed=False, judge_passed=True, reason="<looks fine>"),
        CalibrationItem(case_id="false-fail", human_passed=True, judge_passed=False),
        CalibrationItem(case_id="boom", human_passed=True, error="judge error: bad json"),
    ]
    res = summarize(items, suite_name="s", judge="mock:replay")
    out = write_calibration_report(res, tmp_path / "r" / "cal.html")
    doc = out.read_text(encoding="utf-8")
    assert "Judge calibration" in doc and "Confusion matrix" in doc
    assert doc.index("false-pass") < doc.index("false-fail")  # false passes first
    assert "&lt;looks fine&gt;" in doc  # escaped
    assert "Judge errors" in doc and "bad json" in doc


def test_calibration_report_cards_and_confusion_counts(tmp_path):
    items = [
        CalibrationItem(case_id="tp1", human_passed=True, judge_passed=True, human_score=1.0, judge_score=0.9),
        CalibrationItem(case_id="tp2", human_passed=True, judge_passed=True, human_score=0.8, judge_score=0.8),
        CalibrationItem(case_id="tn", human_passed=False, judge_passed=False, human_score=0.1, judge_score=0.2),
        CalibrationItem(case_id="fp", human_passed=False, judge_passed=True),
    ]
    res = summarize(items, suite_name="cards", judge="j", unlabeled=2)
    doc = write_calibration_report(res, tmp_path / "cal.html").read_text(encoding="utf-8")

    assert _card(doc, "Accuracy") == ("75.0%", "3/4 graded cases agree")
    value, sub = _card(doc, "Cohen&#x27;s kappa")
    assert value == f"{res.kappa:.3f}" and sub == kappa_band(res.kappa)
    assert _card(doc, "False passes") == ("1", "judge passed 1/2 human fails")
    assert _card(doc, "False fails") == ("0", "judge failed 0/2 human passes")
    assert _card(doc, "Score correlation")[1].endswith("3 scored")
    assert _card(doc, "Cases") == ("4", "labeled · 4 graded · 0 errors · 2 unlabeled")
    assert "Judge errors" not in doc

    # Confusion matrix cells: tp, fn on the human-pass row; fp, tn on the human-fail row.
    cells = re.findall(r"<td class='(agree|fn|fp)'><span class='n'>(\d+)</span>", doc)
    assert cells == [("agree", "2"), ("fn", "0"), ("fp", "1"), ("agree", "1")]
    # Every labeled case appears in the full table with its verdict.
    for cid, verdict in [("tp1", "agree"), ("tn", "agree"), ("fp", "disagree")]:
        assert re.search(rf"<code>{cid}</code></td>.*?<td>{verdict}</td></tr>", doc)


def test_calibration_report_includes_case_input_and_expected(tmp_path):
    items = [
        CalibrationItem(case_id="a", human_passed=True, judge_passed=True),
        CalibrationItem(
            case_id="b", human_passed=False, judge_passed=True, output="We refund in 90 days.", reason="ok"
        ),
    ]
    cases = {
        "b": EvalCase(id="b", input="What is the <refund> window?", expected="30 days"),
        "a": EvalCase(id="a", input="unused-input-a"),
    }
    doc = write_calibration_report(summarize(items, suite_name="s", judge="j"), tmp_path / "c.html", cases)
    doc = doc.read_text(encoding="utf-8")
    assert "1 case(s); false passes listed first." in doc
    assert "What is the &lt;refund&gt; window?" in doc and "30 days" in doc and "We refund in 90 days." in doc
    assert "unused-input-a" not in doc  # agreeing cases are not expanded


def test_calibration_report_full_agreement_and_one_class(tmp_path):
    # All human passes, all judged pass: perfect accuracy, but kappa undefined (one class only).
    items = [CalibrationItem(case_id=str(i), human_passed=True, judge_passed=True) for i in range(3)]
    res = summarize(items, suite_name="s", judge="j")
    assert res.kappa is None
    doc = write_calibration_report(res, tmp_path / "c.html").read_text(encoding="utf-8")
    assert "None — the judge agreed with every graded human label." in doc
    assert _card(doc, "Cohen&#x27;s kappa") == ("n/a", "one class only")
    assert _card(doc, "False passes") == ("0", "no human fails")
    assert _card(doc, "Score correlation") == ("n/a", "no human scores")


def test_calibration_report_with_nothing_graded(tmp_path):
    items = [CalibrationItem(case_id="x", human_passed=False, error="adapter error: down")]
    res = summarize(items, suite_name="<suite>", judge="j")
    doc = write_calibration_report(res, tmp_path / "deep" / "dir" / "c.html").read_text(encoding="utf-8")
    assert "<title>Calibration &lt;suite&gt;</title>" in doc
    assert "No graded cases." in doc
    assert _card(doc, "Cohen&#x27;s kappa") == ("n/a", "undefined")
    assert _card(doc, "Accuracy") == ("n/a", "0/0 graded cases agree")
    assert "1 case(s) excluded from the metrics." in doc and "adapter error: down" in doc


def test_kappa_band():
    assert kappa_band(None) == "undefined"
    assert kappa_band(-0.1) == "worse than chance"
    assert kappa_band(0.0) == "slight"
    assert kappa_band(0.2) == "slight"
    assert kappa_band(0.3) == "fair"
    assert kappa_band(0.5) == "moderate"
    assert kappa_band(0.8) == "substantial"
    assert kappa_band(0.95) == "almost perfect"


# --- CLI: report paths and --min-kappa exit codes ------------------------------------------


def _verdict_judge(prompt: str) -> str:
    return json.dumps({"pass": "GOOD" in prompt, "score": 0.9 if "GOOD" in prompt else 0.1, "reason": "fake"})


def _project(tmp_path: Path, monkeypatch, rows: list[dict]) -> None:
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (tmp_path / "suite.yaml").write_text(
        "name: cli calib\ndataset: cases.jsonl\nadapter:\n  type: mock\n  template: '{input}'\n"
        "scorers:\n  - type: llm_judge\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    judge = LLMJudgeScorer(client=FakeOpenAIByPrompt(_verdict_judge))
    monkeypatch.setattr(calibrate_mod, "build_scorer", lambda spec, cache=None: judge)


# kappa = 0.5 for this set: tp=2, fn=1, tn=1 (see test_calibrate_suite_with_labels_file).
ROWS = [
    {"id": "1", "input": "GOOD one", "human_label": {"pass": True}},
    {"id": "2", "input": "GOOD two", "human_label": {"pass": True}},
    {"id": "3", "input": "bad three", "human_label": {"pass": False}},
    {"id": "4", "input": "bad four", "human_label": {"pass": True}},
]


def _invoke(*args: str):
    return CliRunner().invoke(app, ["calibrate", "--suite", "suite.yaml", "--no-cache", *args])


def test_cli_writes_default_report_path(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, ROWS)
    res = _invoke()
    assert res.exit_code == 0, res.output
    report = tmp_path / ".ai-eval" / "reports" / "calibration-cli-calib.html"  # name slugified
    assert report.exists()
    doc = report.read_text(encoding="utf-8")
    assert "Suite: cli calib" in doc
    assert "bad four" in doc  # disagreement shows the case input
    assert "Report:" in res.output


def test_cli_writes_report_to_custom_path(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, ROWS)
    res = _invoke("--report", "out/cal.html")
    assert res.exit_code == 0, res.output
    assert (tmp_path / "out" / "cal.html").exists()
    assert not (tmp_path / ".ai-eval" / "reports").exists()


@pytest.mark.parametrize(
    ("min_kappa", "code"),
    [("0.4", 0), ("0.5", 0), ("0.51", 1), ("1.0", 1), ("-1.0", 0)],
)
def test_cli_min_kappa_exit_codes(tmp_path, monkeypatch, min_kappa, code):
    _project(tmp_path, monkeypatch, ROWS)
    res = _invoke("--min-kappa", min_kappa)
    assert res.exit_code == code, res.output
    assert ("below min" in res.output) == bool(code)
    # The report is written even when the gate fails.
    assert (tmp_path / ".ai-eval" / "reports" / "calibration-cli-calib.html").exists()


def test_cli_min_kappa_fails_when_kappa_undefined(tmp_path, monkeypatch):
    rows = [{"id": str(i), "input": "GOOD", "human_label": {"pass": True}} for i in range(3)]
    _project(tmp_path, monkeypatch, rows)
    assert _invoke().exit_code == 0  # no gate: undefined kappa is reported, not fatal
    res = _invoke("--min-kappa", "-1.0")
    assert res.exit_code == 1
    assert "n/a below min" in res.output


def test_cli_min_kappa_out_of_range_is_usage_error(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, ROWS)
    assert _invoke("--min-kappa", "1.5").exit_code == 2


def test_cli_bad_suite_is_usage_error(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, [{"id": "1", "input": "x"}])  # no human labels at all
    res = _invoke()
    assert res.exit_code == 2
    assert "human-labeled" in res.output


def test_cli_example_suite_report(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    report = tmp_path / "example.html"
    res = CliRunner().invoke(
        app,
        ["calibrate", "--suite", str(SUITE), "--labels", str(LABELS), "--no-cache",
         "--report", str(report), "--min-kappa", "0.5"],
    )
    assert res.exit_code == 0, res.output
    doc = report.read_text(encoding="utf-8")
    assert _card(doc, "Accuracy") == ("80.0%", "16/20 graded cases agree")
    assert _card(doc, "False passes")[0] == "3"
    for cid in ("exchange-size", "ship-express", "international-returns", "warranty-water"):
        assert f"<code>{cid}</code>" in doc
