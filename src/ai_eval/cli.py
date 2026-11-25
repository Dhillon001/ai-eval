from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ai_eval.bench import BENCH_COLUMNS, bench_cells, format_markdown, parse_system_spec, run_bench
from ai_eval.calibrate import CalibrationResult, calibrate_suite
from ai_eval.dataset import load_cases, load_suite
from ai_eval.models import RunExport
from ai_eval.report import (
    compare_exports,
    generate_report,
    regression_gate,
    write_calibration_report,
    write_compare_report,
)
from ai_eval.runner import run_suite
from ai_eval.store import RunStore, load_run_file, save_run_file

app = typer.Typer(no_args_is_help=True, help="Offline evaluation runner for AI systems")
console = Console()


def _project_root() -> Path:
    return Path.cwd()


@app.command()
def run(
    suite: Path = typer.Option(..., "--suite", help="Path to suite YAML"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Override adapter: mock, openai, anthropic, http, mcp"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name for the openai/anthropic/mcp adapter"),
    concurrency: int = typer.Option(1, "--concurrency", "-j", min=1, help="Cases to run in parallel"),
    repeats: int = typer.Option(
        1, "--repeats", "-n", min=1,
        help="Run every case N times; a case passes on a strict majority of its samples",
    ),
    min_pass_rate: float = typer.Option(0.0, "--min-pass-rate", help="Exit 1 if pass rate below this"),
    save_baseline: Optional[Path] = typer.Option(
        None, "--save-baseline", help="Also export this run (summary, outcomes, cases) as JSON to PATH"
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass the judge cache in .ai-eval/cache/ (no reads or writes)"
    ),
):
    """Run an eval suite and store results."""
    root = _project_root()
    suite_path = suite if suite.is_absolute() else root / suite
    opts = {}
    if model:
        opts["model"] = model

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
        console=console, transient=True,
    ) as progress:
        task = progress.add_task("Evaluating", total=None)
        run_id, summary, _ = run_suite(
            suite_path=suite_path,
            project_root=root,
            adapter_name=adapter,
            adapter_options=opts or None,
            concurrency=concurrency,
            on_case_done=lambda _o: progress.advance(task),
            use_cache=not no_cache,
            repeats=repeats,
        )
    report_path = generate_report(root, run_id)

    console.print(f"[green]Run complete[/green] id={run_id}")
    console.print(
        f"Pass rate: {summary.passed}/{summary.total} ({summary.pass_rate:.1%}) | "
        f"mean score: {summary.mean_score:.3f}"
    )
    m = summary.metadata
    if repeats > 1:
        console.print(
            f"Repeats: {repeats} per case | sample pass rate: {m.get('sample_pass_rate', 0):.1%} | "
            f"mean consistency: {m.get('mean_consistency', 0):.1%}"
        )
        flaky = m.get("flaky_cases") or []
        if flaky:
            console.print(f"[yellow]Flaky cases: {len(flaky)}/{summary.total} (mixed outcomes across repeats)[/yellow]")
            for f in flaky:
                verdict = "pass" if f["passed"] else "fail"
                console.print(f"  - {escape(f['case_id'])}: {f['passed_samples']}/{f['repeats']} samples passed ({verdict})")
        else:
            console.print("Flaky cases: 0")
    console.print(
        f"Errors: {m.get('errors', 0)} | latency p50/p95: {m.get('latency_ms_p50', 0):.0f}/"
        f"{m.get('latency_ms_p95', 0):.0f} ms"
        + (f" | tokens in/out: {m['prompt_tokens']}/{m['completion_tokens']}" if m.get("prompt_tokens") else "")
        + (f" | judge cache hits: {m['judge_cache_hits']}" if m.get("judge_cache_hits") else "")
    )
    if m.get("cost_usd") is not None:
        console.print(
            f"Cost: ${m['cost_usd']:.4f} (system ${m.get('cost_usd_system') or 0:.4f}"
            f" + judge ${m.get('cost_usd_judge') or 0:.4f})"
        )
    if m.get("unpriced_models"):
        console.print(
            f"[yellow]No price for: {', '.join(m['unpriced_models'])} (add them under `pricing:` in the suite)[/yellow]"
        )
    console.print(f"Report: {report_path}")
    if save_baseline:
        out = save_baseline if save_baseline.is_absolute() else root / save_baseline
        save_run_file(RunStore.default(root).export_run(run_id), out)
        console.print(f"Saved run file: {out}")

    if summary.pass_rate < min_pass_rate:
        console.print(f"[red]Below min pass rate {min_pass_rate:.1%}[/red]")
        raise typer.Exit(code=1)


@app.command()
def bench(
    suite: Path = typer.Option(..., "--suite", help="Path to suite YAML"),
    adapters: list[str] = typer.Option(
        ..., "--adapter", metavar="ADAPTER[:MODEL]",
        help="System to benchmark, e.g. mock, openai:gpt-4o-mini, anthropic:claude-sonnet-4-5. Repeat for each system.",
    ),
    concurrency: int = typer.Option(1, "--concurrency", "-j", min=1, help="Cases to run in parallel per system"),
    markdown: bool = typer.Option(False, "--markdown", help="Print the comparison as a markdown table"),
    out: Optional[Path] = typer.Option(None, "--out", help="Also write the markdown table to PATH"),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass the judge cache in .ai-eval/cache/ (no reads or writes)"
    ),
):
    """Run one suite against several systems and print a comparison table (best first)."""
    root = _project_root()
    suite_path = suite if suite.is_absolute() else root / suite
    try:
        systems = [parse_system_spec(a) for a in adapters]
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--adapter") from e

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
        console=console, transient=True,
    ) as progress:
        task = progress.add_task("Benchmarking", total=None)
        try:
            result = run_bench(
                suite_path=suite_path,
                project_root=root,
                systems=systems,
                concurrency=concurrency,
                use_cache=not no_cache,
                on_case_done=lambda _o: progress.advance(task),
                on_system_start=lambda s: progress.update(task, description=f"Benchmarking {s.label}"),
            )
        except (ValueError, TypeError) as e:
            # e.g. `--adapter http` on a suite that doesn't configure an http url
            raise typer.BadParameter(f"{type(e).__name__}: {e}", param_hint="--adapter") from e

    md = format_markdown(result)
    if markdown:
        typer.echo(md, nl=False)
    else:
        table = Table(title=f"Benchmark: {result.suite_name}")
        for i, col in enumerate(BENCH_COLUMNS):
            edge = i in (0, len(BENCH_COLUMNS) - 1)
            table.add_column(col, justify="left" if edge else "right", no_wrap=edge)
        for row in result.ranked():
            table.add_row(*bench_cells(row))
        console.print(table)
    if out:
        dest = out if out.is_absolute() else root / out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(md, encoding="utf-8")
        console.print(f"Wrote {dest}")


def _fmt(value: Optional[float], spec: str = ".3f") -> str:
    return "n/a" if value is None else f"{value:{spec}}"


def _print_calibration(res: CalibrationResult) -> None:
    console.print(f"[bold]Judge calibration[/bold]: {escape(res.suite_name)} | judge {escape(res.judge)}")
    console.print(
        f"Labeled cases: {res.labeled} (graded {res.n}, judge errors/skips {len(res.errors)}) | "
        f"unlabeled skipped: {res.unlabeled}"
    )
    console.print(
        f"Accuracy: {_fmt(res.accuracy, '.1%')} | Cohen's kappa: {_fmt(res.kappa)}"
        + ("" if res.kappa is not None or not res.n else " (one class only)")
    )
    cm = res.confusion
    table = Table(title="Confusion matrix (rows: human, cols: judge)")
    table.add_column("")
    table.add_column("judge pass", justify="right")
    table.add_column("judge fail", justify="right")
    table.add_row("human pass", str(cm.tp), str(cm.fn))
    table.add_row("human fail", str(cm.fp), str(cm.tn))
    console.print(table)
    if res.score_pairs:
        console.print(
            f"Score correlation ({res.score_pairs} cases with human scores): "
            f"Pearson {_fmt(res.pearson)} | Spearman {_fmt(res.spearman)}"
        )
    else:
        console.print("Score correlation: n/a (no human scores)")
    for item in res.disagreements:
        verdict = "passed" if item.judge_passed else "failed"
        human = "pass" if item.human_passed else "fail"
        console.print(f"  [yellow]disagree[/yellow] {escape(item.case_id)}: judge {verdict}, human {human}")
    for item in res.errors:
        console.print(f"  [red]error[/red] {escape(item.case_id)}: {escape(item.error or '')}")


@app.command()
def calibrate(
    suite: Path = typer.Option(..., "--suite", help="Path to suite YAML (must have an LLM judge scorer)"),
    labels: Optional[Path] = typer.Option(
        None, "--labels", help="Human labels JSONL keyed by case id (else inline human_label fields)"
    ),
    judge: Optional[str] = typer.Option(
        None, "--judge", help="Name/type of the judge scorer to calibrate (default: first LLM judge)"
    ),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Override adapter: mock, openai, anthropic, http, mcp"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name for the openai/anthropic/mcp adapter"),
    concurrency: int = typer.Option(1, "--concurrency", "-j", min=1, help="Cases to run in parallel"),
    min_kappa: Optional[float] = typer.Option(
        None, "--min-kappa", min=-1.0, max=1.0,
        help="Exit 1 if Cohen's kappa is below this (or undefined)",
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Also write the full result as JSON to PATH"),
    report: Optional[Path] = typer.Option(
        None, "--report",
        help="Where to write the HTML calibration report (default: .ai-eval/reports/calibration-<suite>.html)",
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass the judge cache in .ai-eval/cache/ (no reads or writes)"
    ),
):
    """Measure how well the suite's LLM judge agrees with human labels."""
    root = _project_root()
    suite_path = suite if suite.is_absolute() else root / suite
    labels_path = None if labels is None else (labels if labels.is_absolute() else root / labels)
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
        console=console, transient=True,
    ) as progress:
        task = progress.add_task("Calibrating", total=None)
        try:
            res = calibrate_suite(
                suite_path=suite_path,
                project_root=root,
                labels_path=labels_path,
                adapter_name=adapter,
                adapter_options={"model": model} if model else None,
                judge_name=judge,
                concurrency=concurrency,
                use_cache=not no_cache,
                on_case_done=lambda _i: progress.advance(task),
            )
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e

    _print_calibration(res)
    if report is None:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", res.suite_name).strip("-") or "suite"
        report_path = root / ".ai-eval" / "reports" / f"calibration-{slug}.html"
    else:
        report_path = report if report.is_absolute() else root / report
    try:
        cases = {c.id: c for c in load_cases(Path(load_suite(suite_path, project_root=root).dataset))}
    except Exception:
        cases = {}
    write_calibration_report(res, report_path, cases)
    console.print(f"Report: {report_path}")
    if out:
        dest = out if out.is_absolute() else root / out
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = res.model_dump(mode="json")
        payload["disagreements"] = [i.case_id for i in res.disagreements]
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Wrote {dest}")
    if min_kappa is not None and (res.kappa is None or res.kappa < min_kappa):
        console.print(f"[red]Cohen's kappa {_fmt(res.kappa)} below min {min_kappa:.3f}[/red]")
        raise typer.Exit(code=1)


@app.command("list-runs")
def list_runs(limit: int = typer.Option(20, "--limit")):
    store = RunStore.default(_project_root())
    rows = store.list_runs(limit=limit)
    table = Table(title="Eval runs")
    table.add_column("Run ID")
    table.add_column("Suite")
    table.add_column("Pass rate")
    table.add_column("Adapter")
    table.add_column("Created")
    for r in rows:
        table.add_row(
            r.run_id,
            r.suite_name,
            f"{r.pass_rate:.1%}",
            r.adapter,
            r.created_at.isoformat(),
        )
    console.print(table)


def _load_run_ref(store: RunStore, ref: str) -> RunExport:
    """A run id in the local store, or a path to a run JSON written by `run --save-baseline`."""
    path = Path(ref)
    path = path if path.is_absolute() else _project_root() / path
    try:
        if ref.endswith(".json") or path.is_file():
            return load_run_file(path)
        return store.export_run(ref)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


def _safe_name(ref: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in Path(ref).stem)


@app.command()
def report(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Write the HTML report for one run"),
    compare: Optional[tuple[str, str]] = typer.Option(
        None, "--compare", metavar="A B",
        help="Side-by-side HTML diff of two runs (A = baseline, B = candidate); each is a run id or a run JSON file",
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Output HTML path (default: .ai-eval/reports/)"),
):
    """Write an HTML report for one run, or a side-by-side comparison of two runs."""
    root = _project_root()
    if bool(run_id) == bool(compare):
        raise typer.BadParameter("Pass exactly one of --run-id or --compare A B")
    if run_id:
        path = generate_report(root, run_id)
        if out:
            dest = out if out.is_absolute() else root / out
            dest.parent.mkdir(parents=True, exist_ok=True)
            path = Path(shutil.copyfile(path, dest))
        console.print(f"Wrote {path}")
        return

    a_ref, b_ref = compare  # type: ignore[misc]
    store = RunStore.default(root)
    a, b = _load_run_ref(store, a_ref), _load_run_ref(store, b_ref)
    dest = out or Path(".ai-eval") / "reports" / f"compare_{_safe_name(a_ref)}_vs_{_safe_name(b_ref)}.html"
    path = write_compare_report(a, b, dest if dest.is_absolute() else root / dest)
    diff = compare_exports(a, b)
    console.print(
        f"Regressions: {len(diff['regressions'])} | score drops: {len(diff['score_drops'])} | "
        f"improvements: {len(diff['improvements'])} | pass rate {a.summary.pass_rate:.1%} -> "
        f"{b.summary.pass_rate:.1%}"
    )
    sig = diff["significance"]
    score = sig.get("score_delta")
    if score:
        console.print(
            f"Significance ({sig['n_pairs']} paired cases): McNemar p={sig['mcnemar']['p_value']:.3g} | "
            f"score delta {score['estimate']:+.3f} [{score['low']:+.3f}, {score['high']:+.3f}] "
            f"p={score['p_value']:.3g}"
        )
    console.print(f"Wrote {path}")


def _resolve_run(store: RunStore, run_id: Optional[str], run_file: Optional[Path], which: str) -> RunExport:
    if bool(run_id) == bool(run_file):
        raise typer.BadParameter(f"Pass exactly one of --{which} or --{which}-file")
    try:
        if run_file:
            return load_run_file(run_file if run_file.is_absolute() else _project_root() / run_file)
        return store.export_run(run_id)  # type: ignore[arg-type]
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


@app.command()
def compare(
    baseline: Optional[str] = typer.Option(None, "--baseline", help="Baseline run id in the local store"),
    candidate: Optional[str] = typer.Option(None, "--candidate", help="Candidate run id in the local store"),
    baseline_file: Optional[Path] = typer.Option(
        None, "--baseline-file", help="Baseline run JSON written by `run --save-baseline`"
    ),
    candidate_file: Optional[Path] = typer.Option(
        None, "--candidate-file", help="Candidate run JSON written by `run --save-baseline`"
    ),
    fail_on_score_drop: bool = typer.Option(False, "--fail-on-score-drop", help="Also exit 1 on score drops"),
    alpha: Optional[float] = typer.Option(
        None, "--alpha", min=0.0, max=1.0,
        help="Only fail when the drop is statistically significant (p < ALPHA; McNemar for "
        "pass/fail flips, paired bootstrap for score)",
    ),
    min_effect: Optional[float] = typer.Option(
        None, "--min-effect", min=0.0,
        help="Only fail when the net drop is at least this big (pass rate as a fraction for "
        "regressions, mean score for score drops)",
    ),
):
    """Diff two runs (from the local store or exported JSON files). Exit 1 on regressions.

    With --alpha / --min-effect, regressions that are within noise or too small are reported
    but do not fail; see the "gate" key in the output.
    """
    if alpha is not None and alpha <= 0.0:
        raise typer.BadParameter("--alpha must be in (0, 1]")
    store = RunStore.default(_project_root())
    b = _resolve_run(store, baseline, baseline_file, "baseline")
    c = _resolve_run(store, candidate, candidate_file, "candidate")
    data = compare_exports(b, c)
    data["gate"] = regression_gate(
        data, fail_on_score_drop=fail_on_score_drop, alpha=alpha, min_effect=min_effect
    )
    console.print(json.dumps(data, indent=2))
    if data["gate"]["failed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
