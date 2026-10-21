from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ai_eval.report import generate_report, print_compare_json
from ai_eval.runner import run_suite
from ai_eval.store import RunStore

app = typer.Typer(no_args_is_help=True, help="Offline evaluation runner for AI systems")
console = Console()


def _project_root() -> Path:
    return Path.cwd()


@app.command()
def run(
    suite: Path = typer.Option(..., "--suite", help="Path to suite YAML"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Override adapter: mock, openai, anthropic"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name for the openai/anthropic adapter"),
    concurrency: int = typer.Option(1, "--concurrency", "-j", min=1, help="Cases to run in parallel"),
    min_pass_rate: float = typer.Option(0.0, "--min-pass-rate", help="Exit 1 if pass rate below this"),
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


@app.command()
def compare(
    baseline: str = typer.Option(..., "--baseline"),
    candidate: str = typer.Option(..., "--candidate"),
    fail_on_score_drop: bool = typer.Option(False, "--fail-on-score-drop", help="Also exit 1 on score drops"),
):
    text = print_compare_json(_project_root(), baseline, candidate)
    console.print(text)
    data = json.loads(text)
    if data.get("regressions") or (fail_on_score_drop and data.get("score_drops")):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
