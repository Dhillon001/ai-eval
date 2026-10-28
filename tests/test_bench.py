from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_eval.adapters import build_adapter as real_build_adapter
from ai_eval.adapters.mock import MockAdapter
from ai_eval.bench import (
    BENCH_COLUMNS,
    BenchResult,
    BenchRow,
    SystemSpec,
    format_markdown,
    parse_system_spec,
    run_bench,
)
from ai_eval.cli import app
from ai_eval.store import RunStore

ROOT = Path(__file__).resolve().parents[1]
SUITE = "configs/example_suite.yaml"

runner = CliRunner()


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A throwaway project root with the example suites, so .ai-eval/ lands in tmp."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "datasets", tmp_path / "datasets")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def weak_mock(monkeypatch):
    """`mock:weak` answers unhelpfully (fails the checklist); every other mock behaves as usual."""

    def build(name, options=None):
        if name == "mock" and (options or {}).get("model") == "weak":
            return MockAdapter(template="Sorry, no idea.")
        return real_build_adapter(name, options)

    monkeypatch.setattr("ai_eval.runner.build_adapter", build)


def _row(system: str, pass_rate: float, mean: float, **kw) -> BenchRow:
    return BenchRow(
        system=system, run_id=f"run-{system}", total=10, passed=round(pass_rate * 10),
        pass_rate=pass_rate, mean_score=mean, **kw,
    )


# --- system spec parsing ----------------------------------------------------


@pytest.mark.parametrize(
    "spec, adapter, model, label",
    [
        ("mock", "mock", None, "mock"),
        ("openai:gpt-4o-mini", "openai", "gpt-4o-mini", "openai:gpt-4o-mini"),
        (" anthropic : claude-sonnet-4-5 ", "anthropic", "claude-sonnet-4-5", "anthropic:claude-sonnet-4-5"),
        ("openai_compat:llama3:8b", "openai_compat", "llama3:8b", "openai_compat:llama3:8b"),
        ("claude:claude-haiku-4-5", "claude", "claude-haiku-4-5", "claude:claude-haiku-4-5"),
    ],
)
def test_parse_system_spec(spec, adapter, model, label):
    s = parse_system_spec(spec)
    assert (s.adapter, s.model, s.label) == (adapter, model, label)


@pytest.mark.parametrize(
    "spec, msg",
    [("", "expected ADAPTER"), (":gpt-4o", "expected ADAPTER"), ("nope:x", "Unknown adapter"), ("openai:", "empty")],
)
def test_parse_system_spec_rejects_bad_specs(spec, msg):
    with pytest.raises(ValueError, match=msg):
        parse_system_spec(spec)


# --- run_bench --------------------------------------------------------------


def test_bench_runs_each_system_and_stores_runs(project):
    store = RunStore.default(project)
    started: list[str] = []
    result = run_bench(
        suite_path=project / SUITE,
        project_root=project,
        systems=[parse_system_spec("mock"), parse_system_spec("mock:other")],
        store=store,
        on_system_start=lambda s: started.append(s.label),
    )
    assert started == ["mock", "mock:other"]
    assert result.suite_name == "example-chat-v1"
    assert [r.system for r in result.rows] == ["mock", "mock:other"]
    assert len({r.run_id for r in result.rows}) == 2
    for row in result.rows:
        assert row.total > 0 and row.passed == row.total and row.pass_rate == 1.0
        assert row.errors == 0
        # each system's run is a normal stored run that compare/report can load
        assert store.export_run(row.run_id).summary.run_id == row.run_id


def test_bench_ranks_better_system_first(project, weak_mock):
    result = run_bench(
        suite_path=project / SUITE,
        project_root=project,
        systems=[parse_system_spec("mock:weak"), parse_system_spec("mock")],
    )
    weak, good = result.rows
    assert weak.system == "mock:weak" and weak.pass_rate < good.pass_rate
    assert [r.system for r in result.ranked()] == ["mock", "mock:weak"]


def test_bench_rejects_duplicate_and_empty_systems(project):
    with pytest.raises(ValueError, match="Duplicate system"):
        run_bench(project / SUITE, project, [SystemSpec(adapter="mock"), SystemSpec(adapter="mock")])
    with pytest.raises(ValueError, match="at least one"):
        run_bench(project / SUITE, project, [])


def test_ranking_uses_mean_score_then_given_order():
    result = BenchResult(
        suite_name="s",
        rows=[_row("a", 0.5, 0.9), _row("b", 0.8, 0.1), _row("c", 0.5, 0.95), _row("d", 0.5, 0.9)],
    )
    assert [r.system for r in result.ranked()] == ["b", "c", "a", "d"]


# --- markdown ---------------------------------------------------------------


def test_format_markdown_table():
    result = BenchResult(
        suite_name="rag|v1",
        rows=[_row("mock", 0.5, 0.5), _row("openai:gpt-4o-mini", 0.9, 0.8, cost_usd=0.01234, latency_ms_p50=812.4)],
    )
    md = format_markdown(result)
    lines = md.splitlines()
    assert lines[0] == r"**Suite:** `rag\|v1`"
    assert lines[2] == "| " + " | ".join(BENCH_COLUMNS) + " |"
    assert lines[3].count("|") == len(BENCH_COLUMNS) + 1
    best, worst = lines[4], lines[5]
    assert best.startswith("| `openai:gpt-4o-mini` | 90.0% | 9/10 | 0.800 |")
    assert "| 812 |" in best and "$0.0123" in best and best.endswith("| `run-openai:gpt-4o-mini` |")
    assert worst.startswith("| `mock` | 50.0% |") and "| n/a |" in worst
    assert md.endswith("\n")


def test_format_markdown_escapes_cells():
    result = BenchResult(suite_name="s", rows=[_row("evil|sys\nname", 1.0, 1.0)])
    row = format_markdown(result).splitlines()[4]
    assert r"`evil\|sys name`" in row
    assert row.count("|") - row.count(r"\|") == len(BENCH_COLUMNS) + 1


# --- CLI --------------------------------------------------------------------


def test_cli_bench_table(project, weak_mock):
    res = runner.invoke(app, ["bench", "--suite", SUITE, "--adapter", "mock", "--adapter", "mock:weak"])
    assert res.exit_code == 0, res.output
    assert "Benchmark: example-chat-v1" in res.output
    body = [line for line in res.output.splitlines() if line.startswith("│")]
    assert [line.split("│")[1].strip() for line in body] == ["mock", "mock:weak"]
    assert "3/3" in body[0] and "3/3" not in body[1]
    assert len(RunStore.default(project).list_runs()) == 2


def test_cli_bench_markdown_and_out(project, weak_mock):
    res = runner.invoke(
        app,
        ["bench", "--suite", SUITE, "--adapter", "mock:weak", "--adapter", "mock", "--markdown", "--out", "bench/results.md"],
    )
    assert res.exit_code == 0, res.output
    written = (project / "bench" / "results.md").read_text(encoding="utf-8")
    assert written.startswith("**Suite:** `example-chat-v1`")
    rows = [line for line in written.splitlines() if line.startswith("| `")]
    assert [r.split("|")[1].strip() for r in rows] == ["`mock`", "`mock:weak`"]
    assert written in res.output


@pytest.mark.parametrize(
    "adapters, msg",
    [(["nope"], "Unknown adapter"), (["openai:"], "empty"), (["mock", "mock"], "Duplicate")],
)
def test_cli_bench_bad_adapter_is_usage_error(project, adapters, msg):
    args = ["bench", "--suite", SUITE]
    for a in adapters:
        args += ["--adapter", a]
    res = runner.invoke(app, args)
    assert res.exit_code == 2
    assert msg in res.output
    assert not (project / ".ai-eval" / "runs").exists() or not RunStore.default(project).list_runs()


def test_cli_bench_requires_an_adapter(project):
    res = runner.invoke(app, ["bench", "--suite", SUITE])
    assert res.exit_code == 2
