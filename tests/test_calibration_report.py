from __future__ import annotations

from ai_eval.calibrate import CalibrationItem, summarize
from ai_eval.report import kappa_band, write_calibration_report


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


def test_kappa_band():
    assert kappa_band(None) == "undefined"
    assert kappa_band(-0.1) == "worse than chance"
    assert kappa_band(0.5) == "moderate"
    assert kappa_band(0.95) == "almost perfect"
