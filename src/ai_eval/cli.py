from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ai_eval.models import RunExport
from ai_eval.report import compare_exports, generate_report
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


@app.command()
def report(run_id: str = typer.Option(..., "--run-id")):
    path = generate_report(_project_root(), run_id)
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
