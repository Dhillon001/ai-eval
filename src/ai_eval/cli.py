from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ai_eval.models import RunExport
from ai_eval.report import compare_exports, generate_report, write_compare_report
from ai_eval.runner import run_suite
from ai_eval.store import RunStore, load_run_file, save_run_file

app = typer.Typer(no_args_is_help=True, help="Offline evaluation runner for AI systems")
console = Console()


def _project_root() -> Path:
    return Path.cwd()


@app.command()
def run(
    suite: Path = typer.Option(..., "--suite", help="Path to suite YAML"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Override adapter: mock, openai, anthropic, http"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name for the openai/anthropic adapter"),
    concurrency: int = typer.Option(1, "--concurrency", "-j", min=1, help="Cases to run in parallel"),
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
        )
    report_path = generate_report(root, run_id)

    console.print(f"[green]Run complete[/green] id={run_id}")
    console.print(
        f"Pass rate: {summary.passed}/{summary.total} ({summary.pass_rate:.1%}) | "
        f"mean score: {summary.mean_score:.3f}"
    )
    m = summary.metadata
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
):
    """Diff two runs (from the local store or exported JSON files). Exit 1 on regressions."""
    store = RunStore.default(_project_root())
    b = _resolve_run(store, baseline, baseline_file, "baseline")
    c = _resolve_run(store, candidate, candidate_file, "candidate")
    data = compare_exports(b, c)
    console.print(json.dumps(data, indent=2))
    if data.get("regressions") or (fail_on_score_drop and data.get("score_drops")):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
