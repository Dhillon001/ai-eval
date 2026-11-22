"""Tests for ``ai_eval.stats``: bootstrap confidence intervals for pass rate and mean score."""

from __future__ import annotations

import math

import pytest

from ai_eval.models import CaseOutcome, RunResult, ScoreResult
from ai_eval.runner import merge_samples
from ai_eval.stats import ConfidenceInterval, bootstrap_ci, mean_score_ci, pass_rate_ci, quantile


def _outcome(*scores: float, case_id: str = "c", error: str | None = None) -> CaseOutcome:
    run = RunResult(case_id=case_id, output="", error=error)
    return CaseOutcome(
        case_id=case_id,
        run=run,
        scores=[ScoreResult(scorer=f"s{i}", passed=s >= 0.5, score=s) for i, s in enumerate(scores)],
    )


def _outcomes(passes: int, fails: int) -> list[CaseOutcome]:
    return [_outcome(1.0, case_id=f"p{i}") for i in range(passes)] + [
        _outcome(0.0, case_id=f"f{i}") for i in range(fails)
    ]


# --- quantile ----------------------------------------------------------------------------------


def test_quantile_interpolates():
    vals = [0.0, 1.0, 2.0, 3.0]
    assert quantile(vals, 0.0) == 0.0
    assert quantile(vals, 1.0) == 3.0
    assert quantile(vals, 0.5) == pytest.approx(1.5)
    assert quantile([7.0], 0.3) == 7.0


def test_quantile_rejects_bad_input():
    with pytest.raises(ValueError):
        quantile([], 0.5)
    with pytest.raises(ValueError):
        quantile([1.0], 1.5)


# --- bootstrap_ci ------------------------------------------------------------------------------


def test_bootstrap_is_deterministic_for_a_seed():
    data = [((i * 37) % 101) / 101 for i in range(30)]
    a = bootstrap_ci(data, seed=42, n_resamples=200)
    b = bootstrap_ci(data, seed=42, n_resamples=200)
    assert a == b
    c = bootstrap_ci(data, seed=7, n_resamples=200)
    assert (c.low, c.high) != (a.low, a.high)


def test_bootstrap_brackets_the_mean():
    data = [0.1, 0.4, 0.9, 0.3, 0.7, 0.2, 0.8]
    ci = bootstrap_ci(data)
    assert ci.estimate == pytest.approx(sum(data) / len(data))
    assert ci.low < ci.estimate < ci.high
    assert 0.1 <= ci.low and ci.high <= 0.9
    assert ci.n == len(data) and ci.level == 0.95


def test_constant_data_gives_zero_width_interval():
    ci = bootstrap_ci([1.0] * 10)
    assert ci.estimate == ci.low == ci.high == 1.0
    assert ci.width == 0.0


def test_higher_level_gives_wider_interval():
    data = [float(i % 3) for i in range(30)]
    narrow = bootstrap_ci(data, level=0.8)
    wide = bootstrap_ci(data, level=0.99)
    assert wide.width > narrow.width
    assert wide.low <= narrow.low and narrow.high <= wide.high


def test_more_cases_give_narrower_interval():
    small = bootstrap_ci([1.0, 0.0] * 5)
    large = bootstrap_ci([1.0, 0.0] * 200)
    assert large.width < small.width


def test_custom_statistic():
    ci = bootstrap_ci([1.0, 5.0, 9.0], statistic=max, n_resamples=500)
    assert ci.estimate == 9.0
    assert ci.high == 9.0
    assert ci.low >= 1.0


def test_coverage_is_roughly_nominal():
    """Over many simulated runs of a known Bernoulli(0.7) system, a 90% CI should contain the
    true pass rate most of the time (percentile bootstrap undercovers a little at n=40)."""
    import random

    rng = random.Random(123)
    hits = 0
    trials = 200
    for t in range(trials):
        data = [1.0 if rng.random() < 0.7 else 0.0 for _ in range(40)]
        if bootstrap_ci(data, level=0.9, n_resamples=400, seed=t).contains(0.7):
            hits += 1
    assert 0.8 <= hits / trials <= 0.97


@pytest.mark.parametrize(
    "kwargs",
    [{"n_resamples": 0}, {"level": 0.0}, {"level": 1.0}, {"level": 1.5}],
)
def test_bootstrap_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 0.0], **kwargs)


def test_bootstrap_rejects_empty_input():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_confidence_interval_model():
    ci = ConfidenceInterval(estimate=0.8, low=0.64, high=0.92, n=25)
    assert ci.format() == "0.800 [0.640, 0.920]"
    assert ci.format(".0%") == "80% [64%, 92%]"
    assert ci.contains(0.7) and not ci.contains(0.95)
    assert ci.width == pytest.approx(0.28)
    with pytest.raises(ValueError):
        ConfidenceInterval(estimate=0.5, low=0.9, high=0.1)
    assert ConfidenceInterval.model_validate(ci.model_dump()) == ci


# --- pass_rate_ci ------------------------------------------------------------------------------


def test_pass_rate_ci_matches_summary_estimate():
    outcomes = _outcomes(passes=8, fails=2)
    ci = pass_rate_ci(outcomes)
    assert ci.estimate == pytest.approx(0.8)
    assert ci.n == 10
    assert ci.low < 0.8 < ci.high
    assert 0.0 <= ci.low and ci.high <= 1.0


def test_pass_rate_ci_all_pass_and_all_fail():
    assert pass_rate_ci(_outcomes(5, 0)).model_dump(include={"estimate", "low", "high"}) == {
        "estimate": 1.0, "low": 1.0, "high": 1.0,
    }
    ci = pass_rate_ci(_outcomes(0, 5))
    assert (ci.estimate, ci.low, ci.high) == (0.0, 0.0, 0.0)


def test_pass_rate_ci_empty_run():
    ci = pass_rate_ci([])
    assert (ci.estimate, ci.low, ci.high, ci.n) == (0.0, 0.0, 0.0, 0)


def test_pass_rate_ci_uses_case_majority_with_repeats():
    # Two samples of three pass -> the case passes; resampling is over cases, not samples.
    flaky = merge_samples([_outcome(1.0), _outcome(1.0), _outcome(0.0)]).model_copy(update={"case_id": "x"})
    broken = merge_samples([_outcome(0.0)] * 3).model_copy(update={"case_id": "y"})
    ci = pass_rate_ci([flaky, broken])
    assert ci.n == 2
    assert ci.estimate == pytest.approx(0.5)


# --- mean_score_ci -----------------------------------------------------------------------------


def test_mean_score_ci_pools_scores_like_the_summary():
    # Summary mean_score pools every score: (1 + 0 + 0.5) / 3, not the mean of per-case means.
    outcomes = [_outcome(1.0, 0.0, case_id="a"), _outcome(0.5, case_id="b")]
    ci = mean_score_ci(outcomes)
    assert ci.estimate == pytest.approx(0.5)
    assert ci.n == 2
    assert 0.0 <= ci.low <= ci.estimate <= ci.high <= 1.0


def test_mean_score_ci_includes_every_repeat_sample():
    case = merge_samples([_outcome(1.0), _outcome(0.0), _outcome(0.0), _outcome(0.0)])
    ci = mean_score_ci([case])
    assert ci.estimate == pytest.approx(0.25)
    # One case: every resample is that case, so the interval collapses to the estimate.
    assert ci.low == ci.high == pytest.approx(0.25)


def test_mean_score_ci_falls_back_to_pass_rate_without_scores():
    outcomes = [_outcome(case_id="ok"), _outcome(case_id="bad", error="boom"), _outcome(case_id="ok2")]
    ci = mean_score_ci(outcomes)
    assert ci.estimate == pytest.approx(2 / 3)
    assert ci == pass_rate_ci(outcomes)


def test_mean_score_ci_handles_resamples_without_scores():
    outcomes = [_outcome(0.9, case_id="a"), _outcome(case_id="err", error="boom")]
    ci = mean_score_ci(outcomes, n_resamples=500)
    assert ci.estimate == pytest.approx(0.9)
    assert ci.low == 0.0  # some resamples drew only the score-less case
    assert ci.high == pytest.approx(0.9)
    assert not any(math.isnan(v) for v in (ci.low, ci.high))


def test_mean_score_ci_empty_run():
    assert mean_score_ci([]).n == 0


def test_cis_are_seeded_and_reproducible():
    outcomes = [_outcome(i / 10, case_id=str(i)) for i in range(11)]
    assert mean_score_ci(outcomes, seed=3) == mean_score_ci(outcomes, seed=3)
    assert pass_rate_ci(outcomes, seed=3) == pass_rate_ci(outcomes, seed=3)
