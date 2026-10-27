from __future__ import annotations

import html
import json
from pathlib import Path

from ai_eval.models import CaseOutcome, EvalCase, EvalRunSummary, RunExport
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


def _tag_stats(outcomes: list[CaseOutcome], cases: dict[str, EvalCase]) -> list[dict]:
    """Per-tag counts, pass rate and mean score; cases without tags fall under ``untagged``."""
    buckets: dict[str, list[CaseOutcome]] = {}
    for o in outcomes:
        case = cases.get(o.case_id)
        tags = (case.metadata.get("tags") if case else None) or ["untagged"]
        for t in tags:
            buckets.setdefault(str(t), []).append(o)
    rows = []
    for tag, group in sorted(buckets.items()):
        passed = sum(1 for o in group if o.passed)
        rows.append(
            {
                "tag": tag,
                "total": len(group),
                "passed": passed,
                "pass_rate": passed / len(group),
                "mean_score": sum(_case_score(o) for o in group) / len(group),
                "errors": sum(1 for o in group if o.run.error),
            }
        )
    return rows


def _summary_cards(summary: EvalRunSummary) -> str:
    m = summary.metadata
    cost = m.get("cost_usd")
    cards = [
        ("Pass rate", f"{summary.pass_rate:.1%}", f"{summary.passed}/{summary.total} cases"),
        ("Mean score", f"{summary.mean_score:.3f}", "across all scorers"),
        ("Errors", str(m.get("errors", 0)), "adapter or scorer errors"),
        (
            "Latency p50 / p95",
            f"{m.get('latency_ms_p50', 0):.0f} / {m.get('latency_ms_p95', 0):.0f} ms",
            "per case",
        ),
        (
            "Cost",
            f"${cost:.4f}" if cost is not None else "n/a",
            (
                f"system ${m.get('cost_usd_system') or 0:.4f} + judge ${m.get('cost_usd_judge') or 0:.4f}"
                if cost is not None
                else "no priced model"
            ),
        ),
    ]
    out = []
    for label, value, sub in cards:
        cls = " bad" if label == "Errors" and m.get("errors") else ""
        out.append(
            f"<div class='card{cls}'><div class='label'>{html.escape(label)}</div>"
            f"<div class='value'>{html.escape(value)}</div><div class='sub'>{html.escape(sub)}</div></div>"
        )
    return f"<section class='cards'>{''.join(out)}</section>"


def _tag_table(outcomes: list[CaseOutcome], cases: dict[str, EvalCase]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(r['tag'])}</td><td>{r['passed']}/{r['total']}</td>"
        f"<td>{r['pass_rate']:.1%}</td><td>{r['mean_score']:.3f}</td><td>{r['errors']}</td></tr>"
        for r in _tag_stats(outcomes, cases)
    )
    return (
        "<h2>By tag</h2><table class='tags'><thead><tr><th>Tag</th><th>Passed</th><th>Pass rate</th>"
        f"<th>Mean score</th><th>Errors</th></tr></thead><tbody>{rows}</tbody></table>"
    )


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
    .cards {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1rem 0 1.5rem; }}
    .card {{ border: 1px solid #ccc; border-radius: 8px; padding: 0.75rem 1rem; min-width: 10rem; }}
    .card .label {{ font-size: 0.8rem; color: #555; text-transform: uppercase; letter-spacing: 0.03em; }}
    .card .value {{ font-size: 1.6rem; font-weight: 600; margin: 0.2rem 0; }}
    .card .sub {{ font-size: 0.8rem; color: #666; }}
    .card.bad {{ border-color: #d33; background: #fff0f0; }}
    table.tags {{ width: auto; margin-bottom: 1.5rem; }}
  </style>
</head>
<body>
  <h1>Run {html.escape(summary.run_id)}</h1>
  <p>Suite: {html.escape(summary.suite_name)} | Adapter: {html.escape(summary.adapter)}
  | Model: {html.escape(summary.model or "-")}</p>
  {_summary_cards(summary)}
  {_tag_table(outcomes, cases)}
  <h2>Cases</h2>
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


def compare_exports(baseline: RunExport, candidate: RunExport) -> dict:
    return compare_runs(
        baseline.summary,
        candidate.summary,
        baseline.outcomes,
        candidate.outcomes,
        baseline.cases_by_id(),
        candidate.cases_by_id(),
    )


def print_compare_json(project_root: Path, baseline_id: str, candidate_id: str) -> str:
    store = RunStore.default(project_root)
    if not store.get_run(baseline_id) or not store.get_run(candidate_id):
        raise ValueError("Baseline or candidate run not found")
    result = compare_exports(store.export_run(baseline_id), store.export_run(candidate_id))
    return json.dumps(result, indent=2)
