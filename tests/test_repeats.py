"""Tests for ``--repeats``: sample storage, summary math and concurrency with repeats."""

from __future__ import annotations

import threading
import time
from collections import Counter
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_eval.adapters import mock
from ai_eval.cli import app
from ai_eval.models import CaseOutcome, CaseSample, RunResult, ScoreResult
from ai_eval.pricing import ModelPrice, PriceTable, run_cost
from ai_eval.runner import merge_samples, run_stats, run_suite
from ai_eval.store import RunStore, load_run_file, save_run_file

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs" / "example_suite.yaml"
CASE_IDS = ["refund-1", "greeting-1", "edge-empty-checks"]


def _sample(passed: bool = True, error: str | None = None, latency: float = 1.0, tokens: int | None = None):
    run = RunResult(case_id="c", output="", error=error, latency_ms=latency, prompt_tokens=tokens,
                    completion_tokens=tokens, trace={"model": "m"} if tokens else {})
    return CaseOutcome(case_id="c", run=run, scores=[ScoreResult(scorer="checklist", passed=passed,
                                                                 score=1.0 if passed else 0.0)])


def _repeated(*passes: bool, case_id: str = "c") -> CaseOutcome:
    o = merge_samples([_sample(p) for p in passes])
    return o.model_copy(update={"case_id": case_id})


# --- CaseOutcome math --------------------------------------------------------------------------


def test_single_run_outcome_behaves_as_one_sample():
    ok, bad = _sample(True), _sample(False)
    assert ok.samples == [] and ok.repeats == 1
    assert ok.passed and ok.pass_rate == 1.0 and ok.consistency == 1.0
    assert not bad.passed and bad.pass_rate == 0.0 and bad.consistency == 1.0
    assert len(ok.sample_list()) == 1 and ok.sample_list()[0].run == ok.run


@pytest.mark.parametrize(
    ("passes", "passed", "rate", "consistency"),
    [
        ((True, True, True), True, 1.0, 1.0),
        ((False, False, False), False, 0.0, 1.0),
        ((True, True, False), True, 2 / 3, 2 / 3),  # strict majority passes
        ((True, False, False), False, 1 / 3, 2 / 3),
        ((True, False), False, 0.5, 0.5),  # an even split is not a majority
        ((True, True, False, False), False, 0.5, 0.5),
        ((True, True, True, False), True, 0.75, 0.75),
    ],
)
def test_majority_pass_rate_and_consistency(passes, passed, rate, consistency):
    o = _repeated(*passes)
    assert o.repeats == len(passes) and o.passed_samples == sum(passes)
    assert o.passed is passed
    assert o.pass_rate == pytest.approx(rate)
    assert o.consistency == pytest.approx(consistency)


@pytest.mark.parametrize(
    "passes, flaky",
    [
        ((True,), False),
        ((True, True, True), False),
        ((False, False), False),
        ((True, False, True), True),
        ((False, True), True),
        ((False, False, True), True),
    ],
)
def test_flaky_means_mixed_outcomes(passes, flaky):
    o = _sample(passes[0]) if len(passes) == 1 else _repeated(*passes)
    assert o.flaky is flaky


def test_sample_without_scores_passes_unless_errored():
    assert CaseSample(run=RunResult(case_id="c", output="x")).passed
    assert not CaseSample(run=RunResult(case_id="c", output="", error="boom")).passed


def test_top_level_run_is_ignored_for_verdict_when_samples_exist():
    # The first sample is mirrored at the top; the verdict must come from all samples.
    o = _repeated(False, True, True)
    assert not _sample(False).passed and o.scores[0].passed is False
    assert o.passed


# --- merge_samples -----------------------------------------------------------------------------


def test_merge_samples_requires_one():
    with pytest.raises(ValueError):
        merge_samples([])


def test_merge_single_sample_is_unchanged():
    s = _sample(True)
    assert merge_samples([s]) is s


def test_merge_keeps_first_on_top_and_all_samples_in_order():
    parts = [_sample(True, latency=1), _sample(False, latency=2), _sample(True, latency=3)]
    o = merge_samples(parts)
    assert o.run == parts[0].run and o.scores == parts[0].scores
    assert [s.run.latency_ms for s in o.samples] == [1, 2, 3]
    assert [s.passed for s in o.samples] == [True, False, True]


# --- run_stats / run_cost over samples ---------------------------------------------------------


def test_run_stats_without_repeats_has_no_sample_keys():
    st = run_stats([_sample(True), _sample(False)])
    assert "samples" not in st and "sample_pass_rate" not in st and "mean_consistency" not in st
    assert "flaky_count" not in st and "flaky_cases" not in st


def test_run_stats_covers_every_sample():
    a = merge_samples([_sample(True, latency=10), _sample(True, latency=20), _sample(False, error="x", latency=30)])
    b = merge_samples([_sample(False, latency=40), _sample(False, latency=50), _sample(False, latency=60)])
    st = run_stats([a, b])
    assert st["samples"] == 6
    assert st["errors"] == 1
    assert st["sample_pass_rate"] == pytest.approx(2 / 6, abs=1e-6)
    assert st["mean_consistency"] == pytest.approx((2 / 3 + 1.0) / 2, abs=1e-6)
    assert st["latency_ms_p95"] == 60 and st["latency_ms_p50"] in (30, 40)


def test_run_stats_mixed_repeated_and_single_outcomes():
    st = run_stats([_repeated(True, False, True), _sample(True)])
    assert st["samples"] == 4
    assert st["sample_pass_rate"] == pytest.approx(0.75)
    assert st["mean_consistency"] == pytest.approx((2 / 3 + 1.0) / 2, abs=1e-6)


def test_run_stats_lists_flaky_cases_in_order():
    st = run_stats([
        _repeated(True, False, True, case_id="a"),
        _repeated(True, True, True, case_id="b"),
        _repeated(False, False, True, case_id="c"),
        _repeated(False, False, False, case_id="d"),
        _sample(False),
    ])
    assert st["flaky_count"] == 2
    assert st["flaky_cases"] == [
        {"case_id": "a", "passed_samples": 2, "repeats": 3, "passed": True},
        {"case_id": "c", "passed_samples": 1, "repeats": 3, "passed": False},
    ]


def test_run_stats_no_flaky_cases_when_repeats_agree():
    st = run_stats([_repeated(True, True), _repeated(False, False)])
    assert st["flaky_count"] == 0 and st["flaky_cases"] == []


def test_run_stats_sums_tokens_across_samples():
    o = merge_samples([_sample(True, tokens=5) for _ in range(3)])
    st = run_stats([o])
    assert st["prompt_tokens"] == 15 and st["completion_tokens"] == 15


def test_run_cost_bills_every_sample():
    table = PriceTable({"m": ModelPrice(input_per_mtok=1_000_000.0, output_per_mtok=0.0)})
    single = run_cost([_sample(True, tokens=2)], table)
    repeated = run_cost([merge_samples([_sample(True, tokens=2) for _ in range(3)])], table)
    assert single["cost_usd"] == pytest.approx(2.0)
    assert repeated["cost_usd"] == pytest.approx(6.0)


# --- run_suite with repeats --------------------------------------------------------------------


class _Flaky:
    """Patches MockAdapter.run so chosen calls of a case return output that fails its checklist.

    ``fail_calls[case_id]`` is how many of that case's calls fail. Which calls fail does not
    matter for the verdict, so the outcome is deterministic even when samples run in parallel.
    """

    def __init__(self, monkeypatch, fail_calls: dict[str, int], delay: float = 0.0):
        self.fail_calls = fail_calls
        self.delay = delay
        self.calls: Counter[str] = Counter()
        self.active = self.peak = 0
        self.lock = threading.Lock()
        orig = mock.MockAdapter.run

        def run(adapter, case):
            with self.lock:
                self.calls[case.id] += 1
                n = self.calls[case.id]
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                if self.delay:
                    time.sleep(self.delay)
                if n <= self.fail_calls.get(case.id, 0):
                    return RunResult(case_id=case.id, output="", latency_ms=1.0)
                return orig(adapter, case)
            finally:
                with self.lock:
                    self.active -= 1

        monkeypatch.setattr(mock.MockAdapter, "run", run)


def _run(tmp_path, **kw):
    store = RunStore(tmp_path / "runs.db")
    run_id, summary, outcomes = run_suite(SUITE, ROOT, store=store, use_cache=False, **kw)
    return store, run_id, summary, outcomes


def test_repeats_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="repeats"):
        _run(tmp_path, repeats=0)


def test_repeats_one_stores_no_samples(tmp_path):
    store, run_id, summary, _ = _run(tmp_path, repeats=1)
    assert summary.metadata["repeats"] == 1 and "samples" not in summary.metadata
    assert all(o.samples == [] for o in store.get_outcomes(run_id))


@pytest.mark.parametrize("concurrency", [1, 4])
def test_flaky_cases_majority_and_summary(tmp_path, monkeypatch, concurrency):
    # refund-1 fails 1/3 (passes), greeting-1 fails 2/3 (fails), edge-empty-checks always passes.
    flaky = _Flaky(monkeypatch, {"refund-1": 1, "greeting-1": 2})
    seen: list[str] = []
    _, _, summary, outcomes = _run(tmp_path, repeats=3, concurrency=concurrency,
                                   on_case_done=lambda o: seen.append(o.case_id))

    assert [o.case_id for o in outcomes] == CASE_IDS
    assert Counter(seen) == {cid: 3 for cid in CASE_IDS}  # callback fires once per sample
    assert flaky.calls == {cid: 3 for cid in CASE_IDS}
    by_id = {o.case_id: o for o in outcomes}
    assert by_id["refund-1"].passed and by_id["refund-1"].passed_samples == 2
    assert not by_id["greeting-1"].passed and by_id["greeting-1"].passed_samples == 1
    assert by_id["edge-empty-checks"].consistency == 1.0
    for o in outcomes:
        assert all(s.run.case_id == o.case_id for s in o.samples)  # samples grouped by case

    assert summary.total == 3 and summary.passed == 2
    assert summary.pass_rate == pytest.approx(2 / 3)
    m = summary.metadata
    assert m["repeats"] == 3 and m["samples"] == 9
    assert m["sample_pass_rate"] == pytest.approx(6 / 9, abs=1e-6)
    assert m["mean_consistency"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3, abs=1e-6)
    assert m["flaky_count"] == 2
    assert [f["case_id"] for f in m["flaky_cases"]] == ["refund-1", "greeting-1"]
    assert [o.case_id for o in outcomes if o.flaky] == ["refund-1", "greeting-1"]
    # Mean score averages every sample's scores, not just the first sample's.
    assert summary.mean_score == pytest.approx(
        sum(s.score for o in outcomes for smp in o.samples for s in smp.scores)
        / sum(len(smp.scores) for o in outcomes for smp in o.samples)
    )


def test_repeats_run_in_parallel_and_preserve_order(tmp_path, monkeypatch):
    flaky = _Flaky(monkeypatch, {}, delay=0.05)
    _, _, summary, outcomes = _run(tmp_path, repeats=2, concurrency=4)
    assert flaky.peak > 1
    assert [o.case_id for o in outcomes] == CASE_IDS
    assert all(o.repeats == 2 for o in outcomes)
    assert summary.metadata["errors"] == 0 and summary.metadata["concurrency"] == 4


def test_samples_round_trip_through_store_and_run_file(tmp_path, monkeypatch):
    _Flaky(monkeypatch, {"greeting-1": 2})
    store, run_id, _, outcomes = _run(tmp_path, repeats=3)

    stored = {o.case_id: o for o in store.get_outcomes(run_id)}
    original = {o.case_id: o for o in outcomes}
    assert stored.keys() == original.keys()
    for cid, o in original.items():
        assert stored[cid] == o
        assert stored[cid].passed == o.passed and stored[cid].pass_rate == o.pass_rate
    assert store.get_run(run_id).metadata["samples"] == 9

    path = save_run_file(store.export_run(run_id), tmp_path / "out" / "run.json")
    loaded = {o.case_id: o for o in load_run_file(path).outcomes}
    assert loaded == stored
    assert not loaded["greeting-1"].passed and loaded["greeting-1"].passed_samples == 1


def _cli_project(tmp_path, monkeypatch):
    # The CLI stores runs under the cwd, so run it from a throwaway project copy.
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "s.yaml").write_text(SUITE.read_text(encoding="utf-8"), encoding="utf-8")
    data = tmp_path / "datasets" / "examples"
    data.mkdir(parents=True)
    (data / "chat_v1.jsonl").write_text(
        (ROOT / "datasets" / "examples" / "chat_v1.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)


def test_cli_run_with_repeats(tmp_path, monkeypatch):
    _cli_project(tmp_path, monkeypatch)
    _Flaky(monkeypatch, {"greeting-1": 2})
    result = CliRunner().invoke(app, ["run", "--suite", "configs/s.yaml", "--repeats", "3", "--no-cache"])
    assert result.exit_code == 0, result.output
    assert "Pass rate: 2/3" in result.output
    assert "Repeats: 3 per case" in result.output
    assert "sample pass rate: 77.8%" in result.output  # 7 of 9 samples
    assert "Flaky cases: 1/3" in result.output
    assert "greeting-1: 1/3 samples passed (fail)" in result.output
    run = RunStore.default(tmp_path).list_runs(limit=1)[0]
    assert run.metadata["repeats"] == 3 and run.metadata["samples"] == 9


def test_cli_run_reports_no_flaky_cases(tmp_path, monkeypatch):
    _cli_project(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["run", "--suite", "configs/s.yaml", "--repeats", "2", "--no-cache"])
    assert result.exit_code == 0, result.output
    assert "Flaky cases: 0" in result.output


def test_cli_rejects_zero_repeats():
    result = CliRunner().invoke(app, ["run", "--suite", str(SUITE), "--repeats", "0"])
    assert result.exit_code != 0
