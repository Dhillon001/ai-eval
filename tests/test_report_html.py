"""HTML report structure and escaping of untrusted text (model output, dataset fields, traces)."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from ai_eval.models import (
    CaseOutcome,
    EvalCase,
    EvalRunSummary,
    RunExport,
    RunResult,
    ScoreResult,
    utc_now,
)
from ai_eval.report import write_compare_report, write_html_report

# A payload that would execute or break the page if any field were interpolated raw.
XSS = "<script>alert('x')</script>"
ATTR = "\"'><img src=x onerror=alert(1)>"
UNTRUSTED = f"{XSS}{ATTR}&amp;"


class _Collector(HTMLParser):
    """Records the tags and headings of a document so tests can assert on real structure."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.headings: list[str] = []
        self._in_heading = False

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag in ("h1", "h2"):
            self._in_heading = True
            self.headings.append("")

    def handle_endtag(self, tag):
        if tag in ("h1", "h2"):
            self._in_heading = False

    def handle_data(self, data):
        if self._in_heading:
            self.headings[-1] += data


def _parse(doc: str) -> _Collector:
    p = _Collector()
    p.feed(doc)
    return p


def _assert_no_raw_payload(doc: str) -> None:
    assert "<script" not in doc.lower()
    assert "<img" not in doc.lower()
    assert "onerror=alert(1)>" not in doc  # the attribute-breakout payload never appears unescaped
    assert "&lt;script&gt;" in doc  # ...but the text is still shown, escaped
    assert "&amp;amp;" in doc  # pre-escaped entities are escaped again, not passed through
    tags = set(_parse(doc).tags)
    assert not tags & {"script", "img", "iframe"}


def _summary(run_id: str = "r1", **overrides) -> EvalRunSummary:
    fields = dict(
        run_id=run_id, suite_name="suite", created_at=utc_now(), adapter="mock", model=None,
        config_hash="h", git_sha=None, total=2, passed=1, pass_rate=0.5, mean_score=0.5,
        metadata={"errors": 0, "latency_ms_p50": 5.0, "latency_ms_p95": 9.0},
    )
    fields.update(overrides)
    return EvalRunSummary(**fields)


def _hostile_case_and_outcome(cid: str = "evil") -> tuple[EvalCase, CaseOutcome]:
    case = EvalCase(
        id=cid, input=f"in {UNTRUSTED}", context=f"ctx {UNTRUSTED}", expected=f"exp {UNTRUSTED}",
        metadata={"tags": [f"tag {UNTRUSTED}"]},
    )
    outcome = CaseOutcome(
        case_id=cid,
        run=RunResult(case_id=cid, output=f"out {UNTRUSTED}", trace={"tool_calls": [{"name": UNTRUSTED}]}),
        scores=[
            ScoreResult(
                scorer=f"judge {UNTRUSTED}", passed=False, score=0.2,
                details={"parsed": {"reason": f"why {UNTRUSTED}"}, "note": UNTRUSTED},
            )
        ],
    )
    return case, outcome


def _clean_case_and_outcome(cid: str = "ok") -> tuple[EvalCase, CaseOutcome]:
    case = EvalCase(id=cid, input="2+2?", expected="4", metadata={"tags": ["math"]})
    outcome = CaseOutcome(
        case_id=cid, run=RunResult(case_id=cid, output="4"),
        scores=[ScoreResult(scorer="exact", passed=True, score=1.0)],
    )
    return case, outcome


# --- single-run report -------------------------------------------------------


def test_run_report_has_expected_sections(tmp_path: Path):
    ok_case, ok = _clean_case_and_outcome()
    doc = write_html_report(_summary(), [ok], {"ok": ok_case}, tmp_path / "r.html").read_text()
    parsed = _parse(doc)
    assert doc.startswith("<!DOCTYPE html>") and '<meta charset="utf-8"/>' in doc
    assert parsed.headings == ["Run r1", "By tag", "Cases"]
    assert "<section class='cards'>" in doc
    for label in ("Pass rate", "Mean score", "Errors", "Latency p50 / p95", "Cost"):
        assert f"<div class='label'>{label}</div>" in doc
    assert "<table class='tags'>" in doc and ">math<" in doc
    assert "<details class='case pass'>" in doc and "<code>ok</code>" in doc
    # sections appear in reading order: cards, tag table, then cases
    assert doc.index("class='cards'") < doc.index("<h2>By tag</h2>") < doc.index("<h2>Cases</h2>")


def test_run_report_escapes_untrusted_fields(tmp_path: Path):
    bad_case, bad = _hostile_case_and_outcome(cid=f"id {ATTR}")
    summary = _summary(
        run_id=f"run {XSS}", suite_name=f"suite {XSS}", adapter=f"adp {XSS}", model=f"m {ATTR}",
    )
    doc = write_html_report(summary, [bad], {bad_case.id: bad_case}, tmp_path / "r.html").read_text()
    _assert_no_raw_payload(doc)
    # every untrusted field still reaches the page, in escaped form
    for prefix in ("in ", "ctx ", "exp ", "out ", "why ", "tag ", "judge ", "suite ", "adp "):
        assert f"{prefix}&lt;script&gt;" in doc, prefix
    assert "<title>Eval run &lt;script&gt;" in doc


def test_run_report_escapes_adapter_error(tmp_path: Path):
    case = EvalCase(id="e", input="q")
    outcome = CaseOutcome(case_id="e", run=RunResult(case_id="e", output="", error=f"HTTP 500: {UNTRUSTED}"))
    doc = write_html_report(_summary(), [outcome], {"e": case}, tmp_path / "r.html").read_text()
    _assert_no_raw_payload(doc)
    assert "HTTP 500: &lt;script&gt;" in doc and "<details class='case error' open>" in doc


def test_run_report_stays_well_formed(tmp_path: Path):
    """Hostile text must not open or close elements: tag balance matches a clean report."""
    bad_case, bad = _hostile_case_and_outcome()
    ok_case, ok = _clean_case_and_outcome()
    hostile = write_html_report(_summary(), [bad, ok], {"evil": bad_case, "ok": ok_case}, tmp_path / "a.html")
    bad_case2 = bad_case.model_copy(update={"input": "in", "context": "ctx", "expected": "exp",
                                            "metadata": {"tags": ["tag"]}})
    bad2 = bad.model_copy(deep=True)
    bad2.run.output = "out"
    bad2.run.trace = {"tool_calls": [{"name": "n"}]}
    bad2.scores[0] = bad2.scores[0].model_copy(
        update={"scorer": "judge", "details": {"parsed": {"reason": "why"}, "note": "n"}}
    )
    clean = write_html_report(_summary(), [bad2, ok], {"evil": bad_case2, "ok": ok_case}, tmp_path / "b.html")

    def tag_counts(path: Path) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in _parse(path.read_text()).tags:
            counts[t] = counts.get(t, 0) + 1
        return counts

    assert tag_counts(hostile) == tag_counts(clean)


# --- comparison report -------------------------------------------------------


def _export(run_id: str, outcomes: list[CaseOutcome], cases: list[EvalCase], **summary) -> RunExport:
    passed = sum(o.passed for o in outcomes)
    return RunExport(
        summary=_summary(run_id=run_id, total=len(outcomes), passed=passed,
                         pass_rate=passed / len(outcomes), **summary),
        outcomes=outcomes,
        cases=cases,
    )


def test_compare_report_has_expected_sections(tmp_path: Path):
    ok_case, ok = _clean_case_and_outcome()
    a = _export("runA", [ok], [ok_case])
    b = _export("runB", [ok], [ok_case])
    doc = write_compare_report(a, b, tmp_path / "c.html").read_text()
    assert _parse(doc).headings == ["Run comparison", "By scorer", "By tag", "Cases"]
    assert "<table class='diff cases'>" in doc and "<tr class='unchanged'>" in doc
    assert re.search(r"Regression: 0 \| Score drop: 0 \| Improvement: 0 \| Missing in B: 0 \| "
                     r"New in B: 0 \| Unchanged: 1", doc)
    assert "different suites" not in doc


def test_compare_report_warns_on_different_suites(tmp_path: Path):
    ok_case, ok = _clean_case_and_outcome()
    a = _export("runA", [ok], [ok_case], suite_name="one")
    b = _export("runB", [ok], [ok_case], suite_name="two")
    doc = write_compare_report(a, b, tmp_path / "c.html").read_text()
    assert "the runs are from different suites" in doc


def test_compare_report_escapes_untrusted_fields(tmp_path: Path):
    bad_case, bad_b = _hostile_case_and_outcome()
    bad_a = bad_b.model_copy(deep=True)
    bad_a.scores[0] = bad_a.scores[0].model_copy(update={"passed": True, "score": 1.0})
    gone_case = EvalCase(id=f"gone {ATTR}", input="q")
    gone = CaseOutcome(case_id=gone_case.id, run=RunResult(case_id=gone_case.id, output="", error=XSS))
    a = _export(f"A {XSS}", [bad_a, gone], [bad_case, gone_case], suite_name=f"s {XSS}", model=ATTR)
    b = _export(f"B {XSS}", [bad_b], [bad_case], suite_name=f"s {XSS}")
    doc = write_compare_report(a, b, tmp_path / "c.html").read_text()
    _assert_no_raw_payload(doc)
    assert "<tr class='regression'>" in doc and "<tr class='missing'>" in doc
    for prefix in ("in ", "exp ", "out ", "judge ", "tag ", "A ", "B ", "ERROR: "):
        assert f"{prefix}&lt;script&gt;" in doc, prefix
