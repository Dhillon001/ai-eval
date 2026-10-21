from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
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
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Override adapter: mock, openai"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name for openai adapter"),
    min_pass_rate: float = typer.Option(0.0, "--min-pass-rate", help="Exit 1 if pass rate below this"),
):
    """Run an eval suite and store results."""
    root = _project_root()
    suite_path = suite if suite.is_absolute() else root / suite
    opts = {}
    if model:
        opts["model"] = model

    run_id, summary, _ = run_suite(
        suite_path=suite_path,
        project_root=root,
        adapter_name=adapter,
        adapter_options=opts or None,
    )
    report_path = generate_report(root, run_id)

    console.print(f"[green]Run complete[/green] id={run_id}")
    console.print(
        f"Pass rate: {summary.passed}/{summary.total} ({summary.pass_rate:.1%}) | "
        f"mean score: {summary.mean_score:.3f}"
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
):
    text = print_compare_json(_project_root(), baseline, candidate)
    console.print(text)
    data = json.loads(text)
    if data.get("regressions"):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
