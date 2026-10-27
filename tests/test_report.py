from __future__ import annotations

from pathlib import Path

from ai_eval.models import CaseOutcome, EvalCase, EvalRunSummary, RunResult, ScoreResult, utc_now
from ai_eval.report import _order_failures_first, _tag_stats, write_html_report


def _summary(**metadata) -> EvalRunSummary:
    return EvalRunSummary(
        run_id="r1", suite_name="s", created_at=utc_now(), adapter="mock", model=None,
        config_hash="h", git_sha=None, total=3, passed=1, pass_rate=1 / 3, mean_score=0.5,
        metadata=metadata,
    )


def _fixture():
    cases = {
        "a": EvalCase(id="a", input="q", metadata={"tags": ["rag", "easy"]}),
        "b": EvalCase(id="b", input="q", metadata={"tags": ["rag"]}),
        "c": EvalCase(id="c", input="q"),
    }
    outcomes = [
        CaseOutcome(case_id="a", run=RunResult(case_id="a", output="x"),
                    scores=[ScoreResult(scorer="exact", passed=True, score=1.0)]),
        CaseOutcome(case_id="b", run=RunResult(case_id="b", output="y"),
                    scores=[ScoreResult(scorer="exact", passed=False, score=0.5)]),
        CaseOutcome(case_id="c", run=RunResult(case_id="c", output="", error="boom")),
    ]
    return cases, outcomes


def test_tag_stats_groups_and_untagged():
    cases, outcomes = _fixture()
    stats = {r["tag"]: r for r in _tag_stats(outcomes, cases)}
    assert set(stats) == {"easy", "rag", "untagged"}
    assert stats["rag"]["total"] == 2 and stats["rag"]["passed"] == 1
    assert stats["rag"]["mean_score"] == 0.75
    assert stats["untagged"]["errors"] == 1 and stats["untagged"]["pass_rate"] == 0.0


def test_html_has_summary_cards_and_tag_table(tmp_path: Path):
    cases, outcomes = _fixture()
    summary = _summary(errors=1, latency_ms_p50=12.0, latency_ms_p95=40.0,
                       cost_usd=0.0123, cost_usd_system=0.01, cost_usd_judge=0.0023)
    doc = write_html_report(summary, outcomes, cases, tmp_path / "r.html").read_text()
    for needle in ("Pass rate", "33.3%", "Mean score", "Errors", "12 / 40 ms", "$0.0123", "By tag", ">rag<"):
        assert needle in doc
    assert "card bad" in doc


def test_html_cost_card_without_pricing(tmp_path: Path):
    cases, outcomes = _fixture()
    doc = write_html_report(_summary(), outcomes, cases, tmp_path / "r.html").read_text()
    assert "no priced model" in doc and "card bad" not in doc


def test_failures_first_ordering():
    _, outcomes = _fixture()
    low = CaseOutcome(case_id="d", run=RunResult(case_id="d", output="z"),
                      scores=[ScoreResult(scorer="exact", passed=False, score=0.1)])
    ordered = [o.case_id for o in _order_failures_first(outcomes + [low])]
    assert ordered == ["c", "d", "b", "a"]  # error, lowest-scoring failure, other failure, pass


def test_html_case_details_collapsible(tmp_path: Path):
    cases, outcomes = _fixture()
    outcomes[1].run.trace = {"retrieved_ids": ["doc-7"]}
    outcomes[1].scores.append(
        ScoreResult(scorer="llm_judge", passed=False, score=0.2,
                    details={"parsed": {"reason": "missed the date"}, "judge": "openai:gpt"})
    )
    doc = write_html_report(_summary(), outcomes, cases, tmp_path / "r.html").read_text()
    assert doc.index("<code>c</code>") < doc.index("<code>b</code>") < doc.index("<code>a</code>")
    assert "<details class='case error' open>" in doc
    assert "<details class='case fail' open>" in doc
    assert "<details class='case pass'>" in doc
    assert "Judge:</b> missed the date" in doc and "doc-7" in doc and "Trace</summary>" in doc
