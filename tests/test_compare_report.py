from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ai_eval.cli import app
from ai_eval.models import (
    CaseOutcome,
    EvalCase,
    EvalRunSummary,
    RunExport,
    RunResult,
    ScoreResult,
    utc_now,
)
from ai_eval.report import write_compare_report
from ai_eval.store import save_run_file

runner = CliRunner()


def _outcome(cid: str, passed: bool, score: float) -> CaseOutcome:
    return CaseOutcome(case_id=cid, run=RunResult(case_id=cid, output=f"out-{cid}"),
                       scores=[ScoreResult(scorer="exact", passed=passed, score=score)])


def _export(run_id: str, outcomes: list[CaseOutcome], **metadata) -> RunExport:
    passed = sum(o.passed for o in outcomes)
    return RunExport(
        summary=EvalRunSummary(
            run_id=run_id, suite_name="s", created_at=utc_now(), adapter="mock", model=None,
            config_hash="h", git_sha=None, total=len(outcomes), passed=passed,
            pass_rate=passed / len(outcomes), mean_score=0.5, metadata=metadata,
        ),
        outcomes=outcomes,
        cases=[EvalCase(id=o.case_id, input="q", metadata={"tags": ["rag"]}) for o in outcomes],
    )


def _pair() -> tuple[RunExport, RunExport]:
    a = _export("runA", [_outcome("reg", True, 1.0), _outcome("imp", False, 0.0),
                         _outcome("drop", True, 1.0), _outcome("same", True, 1.0),
                         _outcome("gone", True, 1.0)], cost_usd=0.01)
    b = _export("runB", [_outcome("reg", False, 0.0), _outcome("imp", True, 1.0),
                         _outcome("drop", True, 0.6), _outcome("same", True, 1.0),
                         _outcome("added", True, 1.0)], cost_usd=0.02)
    return a, b


def test_compare_report_orders_and_labels_cases(tmp_path: Path):
    a, b = _pair()
    doc = write_compare_report(a, b, tmp_path / "c.html").read_text()
    order = [doc.index(f"<code>{cid}</code>") for cid in ("reg", "drop", "imp", "gone", "added", "same")]
    assert order == sorted(order)
    assert "<tr class='regression'>" in doc and "<tr class='improvement'>" in doc
    assert "<tr class='score_drop'>" in doc and "Missing in B" in doc and "New in B" in doc
    assert "runA" in doc and "runB" in doc and "By scorer" in doc and "By tag" in doc
    assert "$0.0100 → $0.0200" in doc and "+$0.0100" in doc


def test_cli_report_compare_from_files(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a, b = _pair()
    save_run_file(a, tmp_path / "a.json")
    save_run_file(b, tmp_path / "b.json")
    res = runner.invoke(app, ["report", "--compare", "a.json", "b.json", "--out", "cmp.html"])
    assert res.exit_code == 0, res.output
    assert "Regressions: 1" in res.output
    assert "Run comparison" in (tmp_path / "cmp.html").read_text()


def test_cli_report_requires_exactly_one_mode(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["report"]).exit_code != 0
    assert runner.invoke(app, ["report", "--compare", "missing.json", "b.json"]).exit_code != 0
