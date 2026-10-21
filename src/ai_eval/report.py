from __future__ import annotations

import html
import json
from pathlib import Path

from ai_eval.models import CaseOutcome, EvalCase, EvalRunSummary
from ai_eval.store import RunStore


def compare_runs(
    baseline: EvalRunSummary,
    candidate: EvalRunSummary,
    baseline_outcomes: list[CaseOutcome],
    candidate_outcomes: list[CaseOutcome],
) -> dict:
    b_map = {o.case_id: o for o in baseline_outcomes}
    c_map = {o.case_id: o for o in candidate_outcomes}
    ids = sorted(set(b_map) | set(c_map))

    regressions: list[str] = []
    improvements: list[str] = []
    for cid in ids:
        b = b_map.get(cid)
        c = c_map.get(cid)
        if b and c:
            if b.passed and not c.passed:
                regressions.append(cid)
            elif not b.passed and c.passed:
                improvements.append(cid)

    return {
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        "baseline_pass_rate": baseline.pass_rate,
        "candidate_pass_rate": candidate.pass_rate,
        "pass_rate_delta": candidate.pass_rate - baseline.pass_rate,
        "regressions": regressions,
        "improvements": improvements,
    }


def write_html_report(
    summary: EvalRunSummary,
    outcomes: list[CaseOutcome],
    cases: dict[str, EvalCase],
    out_path: Path,
) -> Path:
    rows = []
    for outcome in outcomes:
        case = cases.get(outcome.case_id)
        inp = html.escape(case.input if case else "")
        out = html.escape(outcome.run.output)
        status = "pass" if outcome.passed else "fail"
        err = html.escape(outcome.run.error or "")
        score_bits = ", ".join(
            f"{s.scorer}: {s.score:.2f} ({'ok' if s.passed else 'fail'})" for s in outcome.scores
        )
        rows.append(
            f"<tr class='{status}'><td>{html.escape(outcome.case_id)}</td>"
            f"<td>{status}</td><td><pre>{inp}</pre></td><td><pre>{out}</pre></td>"
            f"<td>{html.escape(score_bits)}</td><td>{err}</td></tr>"
        )

    doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Eval {html.escape(summary.run_id)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem; vertical-align: top; }}
    tr.fail {{ background: #fff0f0; }}
    tr.pass {{ background: #f6fff6; }}
    pre {{ white-space: pre-wrap; margin: 0; }}
  </style>
</head>
<body>
  <h1>Run {html.escape(summary.run_id)}</h1>
  <p>Suite: {html.escape(summary.suite_name)} | Adapter: {html.escape(summary.adapter)}
  | Pass: {summary.passed}/{summary.total} ({summary.pass_rate:.1%}) | Mean score: {summary.mean_score:.3f}</p>
  <table>
    <thead><tr><th>Case</th><th>Status</th><th>Input</th><th>Output</th><th>Scores</th><th>Error</th></tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def generate_report(project_root: Path, run_id: str) -> Path:
    store = RunStore.default(project_root)
    summary = store.get_run(run_id)
    if not summary:
        raise ValueError(f"Run not found: {run_id}")
    outcomes = store.get_outcomes(run_id)
    cases = store.get_cases(run_id)
    out_path = project_root / ".ai-eval" / "reports" / f"{run_id}.html"
    return write_html_report(summary, outcomes, cases, out_path)


def print_compare_json(project_root: Path, baseline_id: str, candidate_id: str) -> str:
    store = RunStore.default(project_root)
    b = store.get_run(baseline_id)
    c = store.get_run(candidate_id)
    if not b or not c:
        raise ValueError("Baseline or candidate run not found")
    result = compare_runs(
        b,
        c,
        store.get_outcomes(baseline_id),
        store.get_outcomes(candidate_id),
    )
    return json.dumps(result, indent=2)
