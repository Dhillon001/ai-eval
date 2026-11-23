"""Confidence intervals, significance and flaky badges in the run and comparison HTML reports."""

from __future__ import annotations

import re
from pathlib import Path

from ai_eval.models import CaseOutcome, EvalCase, EvalRunSummary, RunExport, RunResult, ScoreResult, utc_now
from ai_eval.report import write_compare_report, write_html_report
from ai_eval.runner import merge_samples
from ai_eval.stats import mean_score_ci, pass_rate_ci


def _single(cid: str, passed: bool) -> CaseOutcome:
    return CaseOutcome(
        case_id=cid,
        run=RunResult(case_id=cid, output="o"),
        scores=[ScoreResult(scorer="exact", passed=passed, score=1.0 if passed else 0.0)],
    )


def _repeated(cid: str, *passes: bool) -> CaseOutcome:
    return merge_samples([_single(cid, p) for p in passes]).model_copy(update={"case_id": cid})


def _export(run_id: str, outcomes: list[CaseOutcome]) -> RunExport:
    passed = sum(o.passed for o in outcomes)
    return RunExport(
        summary=EvalRunSummary(
            run_id=run_id, suite_name="s", created_at=utc_now(), adapter="mock", model=None,
            config_hash="h", git_sha=None, total=len(outcomes), passed=passed,
            pass_rate=passed / len(outcomes), mean_score=0.5, metadata={},
        ),
        outcomes=outcomes,
        cases=[EvalCase(id=o.case_id, input="q") for o in outcomes],
    )


def _run_html(tmp_path: Path, outcomes: list[CaseOutcome]) -> str:
    ex = _export("r1", outcomes)
    return write_html_report(ex.summary, outcomes, ex.cases_by_id(), tmp_path / "r.html").read_text()


# --- run report ---------------------------------------------------------------------------------


def test_run_report_shows_pass_rate_and_score_cis(tmp_path: Path):
    outcomes = [_single(f"c{i}", i % 4 != 0) for i in range(12)]
    doc = _run_html(tmp_path, outcomes)
    pr, ms = pass_rate_ci(outcomes), mean_score_ci(outcomes)
    assert f"95% CI {pr.low:.1%}–{pr.high:.1%}" in doc
    assert f"95% CI {ms.low:.3f}–{ms.high:.3f}" in doc
    assert pr.low < pr.estimate < pr.high  # a real interval, not a point


def test_run_report_without_repeats_has_no_flaky_ui(tmp_path: Path):
    doc = _run_html(tmp_path, [_single("a", True), _single("b", False)])
    assert "Flaky cases" not in doc
    assert "badge flaky" not in doc and "badge samples" not in doc


def test_run_report_flags_flaky_cases(tmp_path: Path):
    outcomes = [
        _repeated("steady", True, True, True),
        _repeated("wobbly", True, False, True),
        _repeated("bad-wobbly", False, True, False),
    ]
    doc = _run_html(tmp_path, outcomes)
    assert re.search(r"<div class='card warn'><div class='label'>Flaky cases</div><div class='value'>2</div>", doc)
    assert "up to 3 repeats" in doc
    assert "flaky 2/3" in doc and "flaky 1/3" in doc
    assert "3/3 samples" in doc  # the consistent repeated case gets a neutral badge
    assert doc.count("<span class='badge flaky'") == 2
    assert "2 case(s) are flaky" in doc
    # The badge sits in the case's own summary line.
    m = re.search(r"<code>wobbly</code>.*?</summary>", doc, re.S)
    assert m and "flaky 2/3" in m.group(0)


def test_run_report_repeats_without_flakes_has_neutral_card(tmp_path: Path):
    doc = _run_html(tmp_path, [_repeated("a", True, True), _repeated("b", False, False)])
    assert "<div class='card'><div class='label'>Flaky cases</div><div class='value'>0</div>" in doc
    assert "badge flaky" not in doc and "are flaky" not in doc


# --- comparison report ------------------------------------------------------------------------


def _compare_html(tmp_path: Path, a: list[CaseOutcome], b: list[CaseOutcome]) -> str:
    return write_compare_report(_export("A", a), _export("B", b), tmp_path / "c.html").read_text()


def _sig_section(doc: str) -> str:
    m = re.search(r"<h2>Significance</h2>.*?</table>", doc, re.S)
    assert m, "missing significance section"
    return m.group(0)


def test_compare_report_flags_significant_regression(tmp_path: Path):
    a = [_single(f"c{i}", True) for i in range(20)]
    b = [_single(f"c{i}", i >= 12) for i in range(20)]  # 12 of 20 cases flip to fail
    sig = _sig_section(_compare_html(tmp_path, a, b))
    assert "Paired tests over 20 case(s)" in sig
    assert "12 regressed / 0 improved" in sig
    assert sig.count("significant regression") == 3  # pass rate, score, McNemar
    assert "&lt;0.001" in sig
    assert "[-" in sig  # CI bounds are signed


def test_compare_report_noise_only_is_not_significant(tmp_path: Path):
    a = [_single(f"c{i}", i not in (0,)) for i in range(20)]
    b = [_single(f"c{i}", i not in (1,)) for i in range(20)]  # one flip each way
    sig = _sig_section(_compare_html(tmp_path, a, b))
    assert "1 regressed / 1 improved" in sig
    assert "significant regression" not in sig and "significant improvement" not in sig
    assert sig.count("not significant") == 3


def test_compare_report_significant_improvement_is_green(tmp_path: Path):
    a = [_single(f"c{i}", False) for i in range(15)]
    b = [_single(f"c{i}", True) for i in range(15)]
    sig = _sig_section(_compare_html(tmp_path, a, b))
    assert "<span class='delta up'>significant improvement" in sig
    assert "significant regression" not in sig


def test_compare_report_shows_per_run_cis(tmp_path: Path):
    a = [_single(f"c{i}", i % 2 == 0) for i in range(10)]
    b = [_single(f"c{i}", True) for i in range(10)]
    doc = _compare_html(tmp_path, a, b)
    pa = pass_rate_ci(a)
    assert f"A 95% CI {pa.low:.1%}–{pa.high:.1%} · B 95% CI 100.0%–100.0%" in doc
    assert doc.count("<div class='ci'>") == 2


def test_compare_report_without_shared_cases(tmp_path: Path):
    doc = _compare_html(tmp_path, [_single("x", True)], [_single("y", True)])
    assert "No cases in common; nothing to test." in doc


def test_compare_report_flaky_badges_per_side(tmp_path: Path):
    a = [_repeated("c", True, True, True), _repeated("d", True, True, True)]
    b = [_repeated("c", True, False, True), _repeated("d", True, True, True)]
    doc = _compare_html(tmp_path, a, b)
    row = re.search(r"<tr class='unchanged'><td class='kind'>Unchanged</td><td><code>c</code>.*?</tr>", doc, re.S)
    assert row and row.group(0).count("flaky 2/3") == 1 and "3/3 samples" in row.group(0)
