from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from ai_eval.adapters import ADAPTERS
from ai_eval.cache import JudgeCache
from ai_eval.models import CaseOutcome, EvalRunSummary
from ai_eval.runner import run_suite
from ai_eval.store import RunStore

_ADAPTER_ALIASES = ("openai_compat", "claude")


class SystemSpec(BaseModel):
    """One system under benchmark, parsed from ``ADAPTER[:MODEL]`` (e.g. ``openai:gpt-4o-mini``)."""

    adapter: str
    model: str | None = None

    @property
    def label(self) -> str:
        return f"{self.adapter}:{self.model}" if self.model else self.adapter


def parse_system_spec(spec: str) -> SystemSpec:
    """Parse ``ADAPTER[:MODEL]``. Only the first ``:`` separates, so model names such as
    ``llama3:8b`` survive intact."""
    adapter, sep, model = spec.strip().partition(":")
    adapter, model = adapter.strip(), model.strip()
    if not adapter:
        raise ValueError(f"Invalid system spec {spec!r}: expected ADAPTER[:MODEL], e.g. openai:gpt-4o-mini")
    if adapter not in ADAPTERS and adapter not in _ADAPTER_ALIASES:
        raise ValueError(f"Unknown adapter {adapter!r} in {spec!r}. Use one of {', '.join(ADAPTERS)}.")
    if sep and not model:
        raise ValueError(f"Invalid system spec {spec!r}: model name after ':' is empty")
    return SystemSpec(adapter=adapter, model=model or None)


class BenchRow(BaseModel):
    """Headline numbers for one system's run of the suite."""

    system: str
    run_id: str
    total: int
    passed: int
    pass_rate: float
    mean_score: float
    errors: int = 0
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    cost_usd: float | None = None

    @classmethod
    def from_summary(cls, system: str, summary: EvalRunSummary) -> BenchRow:
        m = summary.metadata
        return cls(
            system=system,
            run_id=summary.run_id,
            total=summary.total,
            passed=summary.passed,
            pass_rate=summary.pass_rate,
            mean_score=summary.mean_score,
            errors=int(m.get("errors") or 0),
            latency_ms_p50=float(m.get("latency_ms_p50") or 0.0),
            latency_ms_p95=float(m.get("latency_ms_p95") or 0.0),
            cost_usd=m.get("cost_usd"),
        )


class BenchResult(BaseModel):
    suite_name: str
    rows: list[BenchRow]

    def ranked(self) -> list[BenchRow]:
        """Best first: pass rate, then mean score; ties keep the order systems were given."""
        return sorted(self.rows, key=lambda r: (-r.pass_rate, -r.mean_score))


def run_bench(
    suite_path: Path,
    project_root: Path,
    systems: list[SystemSpec],
    store: RunStore | None = None,
    concurrency: int = 1,
    use_cache: bool = True,
    cache: JudgeCache | None = None,
    on_case_done: Callable[[CaseOutcome], None] | None = None,
    on_system_start: Callable[[SystemSpec], None] | None = None,
) -> BenchResult:
    """Run the same suite once per system. Each run is stored like a normal ``ai-eval run``,
    so any two can be diffed afterwards with ``ai-eval compare`` / ``report --compare``."""
    if not systems:
        raise ValueError("bench needs at least one system")
    labels = [s.label for s in systems]
    dupes = sorted({x for x in labels if labels.count(x) > 1})
    if dupes:
        raise ValueError(f"Duplicate system(s): {', '.join(dupes)}")

    if store is None:
        store = RunStore.default(project_root)
    if use_cache and cache is None:
        cache = JudgeCache.default(project_root)

    rows: list[BenchRow] = []
    suite_name = ""
    for spec in systems:
        if on_system_start:
            on_system_start(spec)
        _, summary, _ = run_suite(
            suite_path=suite_path,
            project_root=project_root,
            adapter_name=spec.adapter,
            adapter_options={"model": spec.model} if spec.model else None,
            store=store,
            concurrency=concurrency,
            on_case_done=on_case_done,
            use_cache=use_cache,
            cache=cache,
        )
        suite_name = summary.suite_name
        rows.append(BenchRow.from_summary(spec.label, summary))
    return BenchResult(suite_name=suite_name, rows=rows)


BENCH_COLUMNS = ("System", "Pass rate", "Passed", "Mean score", "Errors", "p50 ms", "p95 ms", "Cost (USD)", "Run ID")


def bench_cells(row: BenchRow) -> list[str]:
    return [
        row.system,
        f"{row.pass_rate:.1%}",
        f"{row.passed}/{row.total}",
        f"{row.mean_score:.3f}",
        str(row.errors),
        f"{row.latency_ms_p50:.0f}",
        f"{row.latency_ms_p95:.0f}",
        "n/a" if row.cost_usd is None else f"${row.cost_usd:.4f}",
        row.run_id,
    ]


def _md_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def format_markdown(result: BenchResult) -> str:
    """GitHub-flavoured markdown table, best system first (ready to paste into a README)."""
    lines = [
        f"**Suite:** `{_md_escape(result.suite_name)}`",
        "",
        "| " + " | ".join(BENCH_COLUMNS) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(BENCH_COLUMNS) - 1)) + "|---|",
    ]
    for row in result.ranked():
        cells = [_md_escape(c) for c in bench_cells(row)]
        cells[0] = f"`{cells[0]}`"
        cells[-1] = f"`{cells[-1]}`"
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
