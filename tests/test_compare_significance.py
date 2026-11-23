"""Significance in ``compare``: a clear regression is flagged, noise is not, and small samples stay
honest (McNemar refuses to call a handful of flips significant)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_eval.cli import app
from ai_eval.models import (
    CaseOutcome,
    CaseSample,
    EvalCase,
    EvalRunSummary,
    RunExport,
    RunResult,
    ScoreResult,
    utc_now,
)
from ai_eval.report import compare_exports
from ai_eval.stats import compare_significance, paired_bootstrap
from ai_eval.store import save_run_file

runner = CliRunner()


def _outcome(cid: str, score: float, *, threshold: float = 0.5) -> CaseOutcome:
    return CaseOutcome(
        case_id=cid,
        run=RunResult(case_id=cid, output=f"out-{cid}"),
        scores=[ScoreResult(scorer="judge", passed=score >= threshold, score=score)],
    )


def _repeated(cid: str, *scores: float) -> CaseOutcome:
    samples = [
        CaseSample(run=RunResult(case_id=cid, output="o"),
                   scores=[ScoreResult(scorer="judge", passed=s >= 0.5, score=s)])
        for s in scores
    ]
    return CaseOutcome(case_id=cid, run=samples[0].run, scores=samples[0].scores, samples=samples)


def _export(run_id: str, outcomes: list[CaseOutcome]) -> RunExport:
    passed = sum(o.passed for o in outcomes)
    return RunExport(
        summary=EvalRunSummary(
            run_id=run_id, suite_name="s", created_at=utc_now(), adapter="mock", model=None,
            config_hash="h", git_sha=None, total=len(outcomes), passed=passed,
            pass_rate=passed / len(outcomes) if outcomes else 0.0, mean_score=0.5,
        ),
        outcomes=outcomes,
        cases=[EvalCase(id=o.case_id, input="q") for o in outcomes],
    )


def _base_scores(n: int) -> list[float]:
    """Deterministic, varied baseline scores in [0.55, 0.95] (all passing)."""
    return [0.55 + 0.4 * ((i * 7) % n) / max(n - 1, 1) for i in range(n)]


# --- clear regression ----------------------------------------------------------------------------


def _clear_regression(n: int = 40, flips: int = 12) -> tuple[list[CaseOutcome], list[CaseOutcome]]:
    base_scores = _base_scores(n)
    base = [_outcome(f"c{i}", s) for i, s in enumerate(base_scores)]
    # `flips` cases collapse to failing; every other case also loses a little.
    cand = [
        _outcome(f"c{i}", 0.1 if i < flips else s - 0.05) for i, s in enumerate(base_scores)
    ]
    return base, cand


def test_clear_regression_is_significant():
    base, cand = _clear_regression()
    sig = compare_significance(base, cand)
    assert sig.n_pairs == 40
    assert (sig.mcnemar.regressions, sig.mcnemar.improvements) == (12, 0)
    assert sig.mcnemar.p_value == pytest.approx(2 / 2**12)
    assert sig.mcnemar.p_value < 0.001
    pr = sig.pass_rate_delta
    assert pr is not None and pr.estimate == pytest.approx(-12 / 40)
    assert pr.high < 0 and pr.p_value < 0.01  # CI excludes zero
    sd = sig.score_delta
    assert sd is not None and sd.estimate < -0.1
    assert sd.low <= sd.estimate <= sd.high < 0 and sd.p_value < 0.01


def test_clear_improvement_is_symmetric():
    base, cand = _clear_regression()
    fwd = compare_significance(base, cand)
    rev = compare_significance(cand, base)
    assert (rev.mcnemar.regressions, rev.mcnemar.improvements) == (0, 12)
    assert rev.mcnemar.p_value == fwd.mcnemar.p_value
    assert rev.score_delta is not None and fwd.score_delta is not None
    assert rev.score_delta.estimate == pytest.approx(-fwd.score_delta.estimate)
    assert rev.score_delta.low > 0 and rev.score_delta.p_value < 0.01


def test_score_only_regression_shows_in_bootstrap_not_mcnemar():
    """Every case still passes but scores sag: no flips, yet the score delta is significant."""
    base_scores = _base_scores(30)
    base = [_outcome(f"c{i}", s) for i, s in enumerate(base_scores)]
    cand = [_outcome(f"c{i}", max(0.5, s - 0.08)) for i, s in enumerate(base_scores)]
    sig = compare_significance(base, cand)
    assert sig.mcnemar.discordant == 0 and sig.mcnemar.p_value == 1.0
    assert sig.pass_rate_delta is not None and sig.pass_rate_delta.p_value == 1.0
    assert sig.score_delta is not None and sig.score_delta.high < 0 and sig.score_delta.p_value < 0.01


# --- noise-only change ---------------------------------------------------------------------------


def _noise(n: int = 40) -> tuple[list[CaseOutcome], list[CaseOutcome]]:
    """Same system, different sample: three cases flip each way and scores jitter symmetrically."""
    base_scores = _base_scores(n)
    base, cand = [], []
    for i, s in enumerate(base_scores):
        jitter = 0.03 if i % 2 else -0.03
        if i < 3:  # pass -> fail
            b, c = s, 0.3
        elif i < 6:  # fail -> pass
            b, c = 0.3, s
        else:
            b, c = s, s + jitter
        base.append(_outcome(f"c{i}", b))
        cand.append(_outcome(f"c{i}", c))
    return base, cand


def test_noise_only_change_is_not_significant():
    base, cand = _noise()
    sig = compare_significance(base, cand)
    assert (sig.mcnemar.regressions, sig.mcnemar.improvements) == (3, 3)
    assert sig.mcnemar.p_value == 1.0
    for delta in (sig.pass_rate_delta, sig.score_delta):
        assert delta is not None
        assert delta.low <= 0.0 <= delta.high  # CI straddles zero
        assert delta.p_value > 0.5
    assert sig.score_delta is not None and abs(sig.score_delta.estimate) < 0.05


def test_noise_still_lists_flips_as_regressions():
    """Significance is reported alongside, not instead of, the per-case diff (exit codes unchanged)."""
    base, cand = _noise()
    diff = compare_exports(_export("A", base), _export("B", cand))
    assert diff["regressions"] == ["c0", "c1", "c2"]
    assert diff["significance"]["mcnemar"]["p_value"] == 1.0


def test_identical_runs_are_exactly_null():
    base, _ = _clear_regression()
    sig = compare_significance(base, base)
    assert sig.mcnemar.p_value == 1.0
    for delta in (sig.pass_rate_delta, sig.score_delta):
        assert delta is not None
        assert delta.estimate == delta.low == delta.high == 0.0 and delta.p_value == 1.0


# --- small samples -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected_p"),
    [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125), (5, 0.0625), (6, 0.03125)],
)
def test_small_samples_need_enough_flips(n, expected_p):
    """Even when *every* case regresses, McNemar needs >= 6 flips to reach p < 0.05."""
    base = [_outcome(f"c{i}", 1.0) for i in range(n)]
    cand = [_outcome(f"c{i}", 0.0) for i in range(n)]
    sig = compare_significance(base, cand)
    assert sig.n_pairs == n
    assert sig.mcnemar.p_value == pytest.approx(expected_p)
    assert (sig.mcnemar.p_value < 0.05) is (n >= 6)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_small_samples_stay_well_formed(n):
    base = [_outcome(f"c{i}", 0.9 - 0.1 * i) for i in range(n)]
    cand = [_outcome(f"c{i}", 0.2 + 0.1 * i) for i in range(n)]
    sig = compare_significance(base, cand)
    for delta in (sig.pass_rate_delta, sig.score_delta):
        assert delta is not None and delta.n == n
        assert all(math.isfinite(v) for v in (delta.estimate, delta.low, delta.high))
        assert delta.low <= delta.estimate <= delta.high
        assert 0.0 < delta.p_value <= 1.0  # +1 correction: never exactly zero


def test_single_pair_bootstrap_is_degenerate():
    d = paired_bootstrap([-0.4])
    assert d.estimate == d.low == d.high == pytest.approx(-0.4)


def test_small_sample_ci_is_wider_than_large_sample_ci():
    """Same 30% regression rate: 10 cases give a much wider interval than 200."""
    def width(n: int) -> float:
        k = n * 3 // 10
        base = [_outcome(f"c{i}", 1.0) for i in range(n)]
        cand = [_outcome(f"c{i}", 0.0 if i < k else 1.0) for i in range(n)]
        d = compare_significance(base, cand).pass_rate_delta
        assert d is not None and d.estimate == pytest.approx(-0.3)
        return d.high - d.low

    assert width(10) > 2 * width(200)


def test_unmatched_cases_are_ignored_and_no_overlap_is_null():
    base = [_outcome("a", 1.0), _outcome("b", 1.0)]
    cand = [_outcome("b", 0.0), _outcome("c", 0.0)]
    sig = compare_significance(base, cand)
    assert sig.n_pairs == 1 and sig.mcnemar.regressions == 1
    none = compare_significance([_outcome("a", 1.0)], [_outcome("z", 0.0)])
    assert none.n_pairs == 0 and none.pass_rate_delta is None and none.score_delta is None
    assert none.mcnemar.p_value == 1.0


# --- repeats, determinism and wiring -------------------------------------------------------------


def test_repeats_use_mean_over_samples():
    base = [_repeated("r", 1.0, 1.0, 1.0)]
    cand = [_repeated("r", 1.0, 0.0, 0.0)]  # majority fails -> a flip
    sig = compare_significance(base, cand)
    assert sig.mcnemar.regressions == 1
    assert sig.score_delta is not None and sig.score_delta.estimate == pytest.approx(-2 / 3)


def test_significance_is_seeded_and_level_aware():
    base, cand = _noise()
    assert compare_significance(base, cand, seed=11) == compare_significance(base, cand, seed=11)
    narrow = compare_significance(base, cand, level=0.5).score_delta
    wide = compare_significance(base, cand, level=0.99).score_delta
    assert narrow is not None and wide is not None
    assert wide.high - wide.low > narrow.high - narrow.low


def test_cli_compare_json_includes_significance(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base, cand = _clear_regression()
    save_run_file(_export("A", base), tmp_path / "a.json")
    save_run_file(_export("B", cand), tmp_path / "b.json")
    res = runner.invoke(app, ["compare", "--baseline-file", "a.json", "--candidate-file", "b.json"])
    assert res.exit_code == 1, res.output
    sig = json.loads(res.output)["significance"]
    assert sig["n_pairs"] == 40
    assert sig["mcnemar"]["regressions"] == 12 and sig["mcnemar"]["p_value"] < 0.001
    assert sig["score_delta"]["high"] < 0
