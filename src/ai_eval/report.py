from __future__ import annotations

import html
import json
from pathlib import Path

from ai_eval.models import CaseOutcome, EvalCase, EvalRunSummary, RunExport, ScoreResult
from ai_eval.store import RunStore


_BASE_CSS = """\
    body { font-family: system-ui, sans-serif; margin: 1.5rem; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 0.5rem; vertical-align: top; }
    tr.fail { background: #fff0f0; }
    tr.pass { background: #f6fff6; }
    details.case { border: 1px solid #ccc; border-left-width: 6px; border-radius: 6px; margin: 0.5rem 0; }
    details.case.pass { border-left-color: #2a2; }
    details.case.fail { border-left-color: #d33; }
    details.case.error { border-left-color: #a0a; }
    details.case > summary { cursor: pointer; padding: 0.5rem 0.75rem; }
    details.case .body { padding: 0 0.75rem 0.75rem; }
    .status { font-weight: 600; text-transform: uppercase; font-size: 0.8rem; }
    .mean, .lat { color: #555; font-size: 0.85rem; margin-left: 0.5rem; }
    .badge { font-size: 0.75rem; border-radius: 4px; padding: 0.1rem 0.4rem; margin-left: 0.25rem; }
    .badge.pass { background: #e3f6e3; }
    .badge.fail { background: #fbe0e0; }
    .field { margin: 0.5rem 0; }
    .flabel { font-size: 0.8rem; color: #555; text-transform: uppercase; }
    .field pre { background: #f7f7f7; padding: 0.4rem; border-radius: 4px; }
    table.scores { margin: 0.5rem 0; }
    .reason { margin-bottom: 0.25rem; }
    pre { white-space: pre-wrap; margin: 0; }
    .cards { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1rem 0 1.5rem; }
    .card { border: 1px solid #ccc; border-radius: 8px; padding: 0.75rem 1rem; min-width: 10rem; }
    .card .label { font-size: 0.8rem; color: #555; text-transform: uppercase; letter-spacing: 0.03em; }
    .card .value { font-size: 1.6rem; font-weight: 600; margin: 0.2rem 0; }
    .card .sub { font-size: 0.8rem; color: #666; }
    .card.bad { border-color: #d33; background: #fff0f0; }
    table.tags { width: auto; margin-bottom: 1.5rem; }
"""


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


def _order_failures_first(outcomes: list[CaseOutcome]) -> list[CaseOutcome]:
    """Errors first, then other failures (lowest score first), then passes in original order."""
    def key(o: CaseOutcome) -> tuple:
        if o.passed:
            return (2, 0.0)
        return (0 if o.run.error else 1, _case_score(o))

    return sorted(outcomes, key=key)


def _judge_reason(s: ScoreResult) -> str | None:
    parsed = s.details.get("parsed")
    if isinstance(parsed, dict) and parsed.get("reason"):
        return str(parsed["reason"])
    return None


def _pre(label: str, text: str | None) -> str:
    if not text:
        return ""
    return f"<div class='field'><div class='flabel'>{html.escape(label)}</div><pre>{html.escape(text)}</pre></div>"


def _score_table(scores: list[ScoreResult]) -> str:
    if not scores:
        return ""
    rows = []
    for s in scores:
        reason = _judge_reason(s)
        extra = {k: v for k, v in s.details.items() if k not in ("raw", "parsed")}
        detail_bits = []
        if reason:
            detail_bits.append(f"<div class='reason'><b>Judge:</b> {html.escape(reason)}</div>")
        if extra:
            detail_bits.append(f"<pre>{html.escape(json.dumps(extra, indent=2, default=str))}</pre>")
        rows.append(
            f"<tr class='{'pass' if s.passed else 'fail'}'><td>{html.escape(s.scorer)}</td>"
            f"<td>{'pass' if s.passed else 'fail'}</td><td>{s.score:.2f}</td>"
            f"<td>{''.join(detail_bits)}</td></tr>"
        )
    return (
        "<table class='scores'><thead><tr><th>Scorer</th><th>Result</th><th>Score</th>"
        f"<th>Details</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _case_details(outcome: CaseOutcome, case: EvalCase | None) -> str:
    status = "error" if outcome.run.error else ("pass" if outcome.passed else "fail")
    badges = " ".join(
        f"<span class='badge {'pass' if s.passed else 'fail'}'>{html.escape(s.scorer)} {s.score:.2f}</span>"
        for s in outcome.scores
    )
    body = [
        _pre("Error", outcome.run.error),
        _pre("Input", case.input if case else None),
        _pre("Context", case.context if case else None),
        _pre("Expected", case.expected if case else None),
        _pre("Output", outcome.run.output),
        _score_table(outcome.scores),
    ]
    if outcome.run.trace:
        body.append(
            "<details class='trace'><summary>Trace</summary>"
            f"<pre>{html.escape(json.dumps(outcome.run.trace, indent=2, default=str))}</pre></details>"
        )
    open_attr = "" if outcome.passed else " open"
    return (
        f"<details class='case {status}'{open_attr}><summary>"
        f"<span class='status'>{status}</span> <code>{html.escape(outcome.case_id)}</code>"
        f" <span class='mean'>score {_case_score(outcome):.2f}</span>"
        f" <span class='lat'>{outcome.run.latency_ms:.0f} ms</span> {badges}</summary>"
        f"<div class='body'>{''.join(body)}</div></details>"
    )


def write_html_report(
    summary: EvalRunSummary,
    outcomes: list[CaseOutcome],
    cases: dict[str, EvalCase],
    out_path: Path,
) -> Path:
    ordered = _order_failures_first(outcomes)
    n_fail = sum(1 for o in outcomes if not o.passed)
    case_blocks = "".join(_case_details(o, cases.get(o.case_id)) for o in ordered)

    doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Eval {html.escape(summary.run_id)}</title>
  <style>
{_BASE_CSS}  </style>
</head>
<body>
  <h1>Run {html.escape(summary.run_id)}</h1>
  <p>Suite: {html.escape(summary.suite_name)} | Adapter: {html.escape(summary.adapter)}
  | Model: {html.escape(summary.model or "-")}</p>
  {_summary_cards(summary)}
  {_tag_table(outcomes, cases)}
  <h2>Cases</h2>
  <p>{n_fail} failing case(s) shown first and expanded; passing cases are collapsed.</p>
  {case_blocks}
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


# --- side-by-side comparison report -----------------------------------------

_COMPARE_CSS = """\
    .card.good { border-color: #2a2; background: #f6fff6; }
    .delta { font-size: 0.85rem; font-weight: 600; }
    .delta.up { color: #1a7f1a; }
    .delta.down { color: #c22; }
    .delta.flat { color: #777; }
    table.diff td.num { text-align: right; white-space: nowrap; }
    tr.regression { background: #fff0f0; }
    tr.score_drop { background: #fff8ec; }
    tr.improvement { background: #f0fff0; }
    tr.missing, tr.new { background: #f4f4f8; }
    .kind { font-weight: 600; text-transform: uppercase; font-size: 0.75rem; white-space: nowrap; }
    .side { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
    .side pre { background: #f7f7f7; padding: 0.4rem; border-radius: 4px; }
    details.out > summary { cursor: pointer; color: #555; font-size: 0.85rem; }
"""

# Display order for case rows: what needs attention first.
_DIFF_KINDS = ("regression", "score_drop", "improvement", "missing", "new", "unchanged")
_DIFF_LABELS = {
    "regression": "Regression",
    "score_drop": "Score drop",
    "improvement": "Improvement",
    "missing": "Missing in B",
    "new": "New in B",
    "unchanged": "Unchanged",
}


def _fmt_delta(
    delta: float | None, fmt: str, *, higher_is_better: bool = True, prefix: str = "", unit: str = ""
) -> str:
    """A signed, colour-coded delta span; ``fmt`` is applied to the absolute value."""
    if delta is None:
        return "<span class='delta flat'>n/a</span>"
    if abs(delta) < 1e-9:
        return "<span class='delta flat'>±0</span>"
    good = (delta > 0) == higher_is_better
    sign = "+" if delta > 0 else "−"
    return f"<span class='delta {'up' if good else 'down'}'>{sign}{prefix}{format(abs(delta), fmt)}{unit}</span>"


def _compare_cards(a: EvalRunSummary, b: EvalRunSummary) -> str:
    am, bm = a.metadata, b.metadata

    def num(m: dict, key: str) -> float | None:
        v = m.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    def diff(x: float | None, y: float | None) -> float | None:
        return None if x is None or y is None else y - x

    a_err, b_err = float(am.get("errors", 0)), float(bm.get("errors", 0))
    a_cost, b_cost = num(am, "cost_usd"), num(bm, "cost_usd")
    rows = [
        ("Pass rate", f"{a.pass_rate:.1%}", f"{b.pass_rate:.1%}",
         _fmt_delta((b.pass_rate - a.pass_rate) * 100, ".1f", unit=" pts"), b.pass_rate - a.pass_rate),
        ("Mean score", f"{a.mean_score:.3f}", f"{b.mean_score:.3f}",
         _fmt_delta(b.mean_score - a.mean_score, ".3f"), b.mean_score - a.mean_score),
        ("Errors", f"{a_err:.0f}", f"{b_err:.0f}",
         _fmt_delta(b_err - a_err, ".0f", higher_is_better=False), a_err - b_err),
    ]
    for key, label in (("latency_ms_p50", "Latency p50"), ("latency_ms_p95", "Latency p95")):
        x, y = num(am, key), num(bm, key)
        rows.append((
            label, f"{x:.0f} ms" if x is not None else "n/a", f"{y:.0f} ms" if y is not None else "n/a",
            _fmt_delta(diff(x, y), ".0f", higher_is_better=False, unit=" ms"), 0.0,
        ))
    rows.append((
        "Cost", f"${a_cost:.4f}" if a_cost is not None else "n/a",
        f"${b_cost:.4f}" if b_cost is not None else "n/a",
        _fmt_delta(diff(a_cost, b_cost), ".4f", higher_is_better=False, prefix="$"),
        0.0,
    ))
    out = []
    for label, av, bv, delta_html, goodness in rows:
        cls = " bad" if goodness < -1e-9 else (" good" if goodness > 1e-9 else "")
        out.append(
            f"<div class='card{cls}'><div class='label'>{html.escape(label)}</div>"
            f"<div class='value'>{html.escape(av)} → {html.escape(bv)}</div>"
            f"<div class='sub'>{delta_html}</div></div>"
        )
    return f"<section class='cards'>{''.join(out)}</section>"


def _case_diff_rows(baseline: RunExport, candidate: RunExport, diff: dict) -> list[dict]:
    a_map = {o.case_id: o for o in baseline.outcomes}
    b_map = {o.case_id: o for o in candidate.outcomes}
    kind_of: dict[str, str] = {cid: "regression" for cid in diff["regressions"]}
    kind_of.update({d["case_id"]: "score_drop" for d in diff["score_drops"]})
    kind_of.update({cid: "improvement" for cid in diff["improvements"]})
    kind_of.update({cid: "missing" for cid in diff["missing_in_candidate"]})
    kind_of.update({cid: "new" for cid in diff["new_in_candidate"]})
    rows = []
    for cid in set(a_map) | set(b_map):
        a, b = a_map.get(cid), b_map.get(cid)
        sa = _case_score(a) if a else None
        sb = _case_score(b) if b else None
        rows.append({
            "case_id": cid,
            "kind": kind_of.get(cid, "unchanged"),
            "a": a,
            "b": b,
            "delta": sb - sa if sa is not None and sb is not None else None,
        })
    rank = {k: i for i, k in enumerate(_DIFF_KINDS)}
    rows.sort(key=lambda r: (rank[r["kind"]], r["delta"] if r["delta"] is not None else 0.0, r["case_id"]))
    return rows


def _side_cell(o: CaseOutcome | None) -> str:
    if o is None:
        return "<td class='num'>—</td><td class='num'>—</td>"
    status = "error" if o.run.error else ("pass" if o.passed else "fail")
    return f"<td class='num'>{status}</td><td class='num'>{_case_score(o):.2f}</td>"


def _side_output(label: str, o: CaseOutcome | None) -> str:
    if o is None:
        return f"<div><div class='flabel'>{html.escape(label)}</div><pre>(not in this run)</pre></div>"
    text = o.run.output if not o.run.error else f"ERROR: {o.run.error}"
    badges = " ".join(
        f"<span class='badge {'pass' if s.passed else 'fail'}'>{html.escape(s.scorer)} {s.score:.2f}</span>"
        for s in o.scores
    )
    return (
        f"<div><div class='flabel'>{html.escape(label)} {badges}</div>"
        f"<pre>{html.escape(text or '')}</pre></div>"
    )


def _case_diff_table(rows: list[dict], cases: dict[str, EvalCase]) -> str:
    body = []
    for r in rows:
        case = cases.get(r["case_id"])
        details = (
            "<details class='out'><summary>outputs</summary>"
            + _pre("Input", case.input if case else None)
            + _pre("Expected", case.expected if case else None)
            + f"<div class='side'>{_side_output('A', r['a'])}{_side_output('B', r['b'])}</div></details>"
        )
        body.append(
            f"<tr class='{r['kind']}'><td class='kind'>{_DIFF_LABELS[r['kind']]}</td>"
            f"<td><code>{html.escape(r['case_id'])}</code>{details}</td>"
            f"{_side_cell(r['a'])}{_side_cell(r['b'])}"
            f"<td class='num'>{_fmt_delta(r['delta'], '.2f')}</td></tr>"
        )
    return (
        "<table class='diff cases'><thead><tr><th>Change</th><th>Case</th><th>A</th><th>A score</th>"
        f"<th>B</th><th>B score</th><th>Δ score</th></tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _scorer_diff_table(baseline: RunExport, candidate: RunExport) -> str:
    am, bm = _scorer_means(baseline.outcomes), _scorer_means(candidate.outcomes)
    rows = []
    for k in sorted(set(am) | set(bm)):
        a, b = am.get(k), bm.get(k)
        rows.append(
            f"<tr><td>{html.escape(k)}</td>"
            f"<td class='num'>{'—' if a is None else f'{a:.3f}'}</td>"
            f"<td class='num'>{'—' if b is None else f'{b:.3f}'}</td>"
            f"<td class='num'>{_fmt_delta(None if a is None or b is None else b - a, '.3f')}</td></tr>"
        )
    return (
        "<h2>By scorer</h2><table class='diff tags'><thead><tr><th>Scorer</th><th>A mean</th>"
        f"<th>B mean</th><th>Δ</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _tag_diff_table(baseline: RunExport, candidate: RunExport) -> str:
    a_stats = {r["tag"]: r for r in _tag_stats(baseline.outcomes, baseline.cases_by_id())}
    b_stats = {r["tag"]: r for r in _tag_stats(candidate.outcomes, candidate.cases_by_id())}

    def cell(r: dict | None) -> str:
        return "<td class='num'>—</td>" if r is None else (
            f"<td class='num'>{r['passed']}/{r['total']} ({r['pass_rate']:.1%})</td>"
        )

    rows = []
    for tag in sorted(set(a_stats) | set(b_stats)):
        a, b = a_stats.get(tag), b_stats.get(tag)
        delta = None if a is None or b is None else (b["pass_rate"] - a["pass_rate"]) * 100
        rows.append(
            f"<tr><td>{html.escape(tag)}</td>{cell(a)}{cell(b)}"
            f"<td class='num'>{_fmt_delta(delta, '.1f', unit=' pts')}</td></tr>"
        )
    return (
        "<h2>By tag</h2><table class='diff tags'><thead><tr><th>Tag</th><th>A passed</th>"
        f"<th>B passed</th><th>Δ pass rate</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _run_label(s: EvalRunSummary) -> str:
    return f"{s.run_id} · {s.adapter}" + (f" / {s.model}" if s.model else "") + f" · {s.suite_name}"


def write_compare_report(baseline: RunExport, candidate: RunExport, out_path: Path) -> Path:
    """Side-by-side HTML diff of two runs: A = ``baseline``, B = ``candidate``.

    Cases are listed regressions first, then score drops, improvements, cases present in only
    one run, and finally unchanged cases. Classification matches ``ai-eval compare``.
    """
    a, b = baseline.summary, candidate.summary
    diff = compare_exports(baseline, candidate)
    rows = _case_diff_rows(baseline, candidate, diff)
    cases = {**baseline.cases_by_id(), **candidate.cases_by_id()}
    counts = {k: sum(1 for r in rows if r["kind"] == k) for k in _DIFF_KINDS}
    count_line = " | ".join(f"{_DIFF_LABELS[k]}: {counts[k]}" for k in _DIFF_KINDS)
    warn = (
        "<p><b>Note:</b> the runs are from different suites; only cases with matching ids are compared.</p>"
        if a.suite_name != b.suite_name
        else ""
    )

    doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Compare {html.escape(a.run_id)} vs {html.escape(b.run_id)}</title>
  <style>
{_BASE_CSS}{_COMPARE_CSS}  </style>
</head>
<body>
  <h1>Run comparison</h1>
  <p><b>A (baseline):</b> {html.escape(_run_label(a))}<br/>
  <b>B (candidate):</b> {html.escape(_run_label(b))}</p>
  {warn}
  {_compare_cards(a, b)}
  <p class='counts'>{count_line}</p>
  {_scorer_diff_table(baseline, candidate)}
  {_tag_diff_table(baseline, candidate)}
  <h2>Cases</h2>
  {_case_diff_table(rows, cases)}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
