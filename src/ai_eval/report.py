from __future__ import annotations

import html
import json
from pathlib import Path

from ai_eval.models import CaseOutcome, EvalCase, EvalRunSummary
from ai_eval.store import RunStore


def _case_score(o: CaseOutcome) -> float:
    if not o.scores:
        return 1.0 if o.run.error is None else 0.0
    return sum(s.score for s in o.scores) / len(o.scores)


def _scorer_means(outcomes: list[CaseOutcome]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for o in outcomes:
        for s in o.scores:
            buckets.setdefault(s.scorer, []).append(s.score)
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def _tag_pass_rates(outcomes: list[CaseOutcome], cases: dict[str, EvalCase]) -> dict[str, float]:
    buckets: dict[str, list[bool]] = {}
    for o in outcomes:
        case = cases.get(o.case_id)
        tags = (case.metadata.get("tags") if case else None) or ["untagged"]
        for t in tags:
            buckets.setdefault(str(t), []).append(o.passed)
    return {t: sum(v) / len(v) for t, v in sorted(buckets.items())}


def compare_runs(
    baseline: EvalRunSummary,
    candidate: EvalRunSummary,
    baseline_outcomes: list[CaseOutcome],
    candidate_outcomes: list[CaseOutcome],
    baseline_cases: dict[str, EvalCase] | None = None,
    candidate_cases: dict[str, EvalCase] | None = None,
    score_drop_threshold: float = 0.1,
) -> dict:
    """Diff two runs. A regression is a case that flipped pass->fail, or (still passing/failing)
    whose mean score dropped by more than `score_drop_threshold`."""
    b_map = {o.case_id: o for o in baseline_outcomes}
    c_map = {o.case_id: o for o in candidate_outcomes}

    regressions: list[str] = []
    improvements: list[str] = []
    score_drops: list[dict] = []
    for cid in sorted(set(b_map) & set(c_map)):
        b, c = b_map[cid], c_map[cid]
        if b.passed and not c.passed:
            regressions.append(cid)
        elif not b.passed and c.passed:
            improvements.append(cid)
        else:
            delta = _case_score(c) - _case_score(b)
            if delta < -score_drop_threshold:
                score_drops.append({"case_id": cid, "delta": round(delta, 4)})

    b_means, c_means = _scorer_means(baseline_outcomes), _scorer_means(candidate_outcomes)
    scorer_deltas = {
        k: round(c_means.get(k, 0.0) - b_means.get(k, 0.0), 4) for k in sorted(set(b_means) | set(c_means))
    }

    result = {
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        "baseline_pass_rate": baseline.pass_rate,
        "candidate_pass_rate": candidate.pass_rate,
        "pass_rate_delta": candidate.pass_rate - baseline.pass_rate,
        "mean_score_delta": candidate.mean_score - baseline.mean_score,
        "regressions": regressions,
        "score_drops": score_drops,
        "improvements": improvements,
        "missing_in_candidate": sorted(set(b_map) - set(c_map)),
        "new_in_candidate": sorted(set(c_map) - set(b_map)),
        "scorer_mean_deltas": scorer_deltas,
    }
    if baseline_cases is not None and candidate_cases is not None:
        b_tags = _tag_pass_rates(baseline_outcomes, baseline_cases)
        c_tags = _tag_pass_rates(candidate_outcomes, candidate_cases)
        result["tag_pass_rate_deltas"] = {
            t: round(c_tags.get(t, 0.0) - b_tags.get(t, 0.0), 4) for t in sorted(set(b_tags) | set(c_tags))
        }
    return result


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
        store.get_cases(baseline_id),
        store.get_cases(candidate_id),
    )
    return json.dumps(result, indent=2)
