"""Bootstrap confidence intervals for run-level metrics.

Eval suites are small samples of the inputs a system will see, so a pass rate of 80% on 25 cases
says less than it seems. These helpers attach a percentile-bootstrap confidence interval to the
headline numbers by resampling *cases* with replacement (a case's repeats stay together, since
they are not independent of each other). Resampling is seeded, so the same run always yields the
same interval. Pure Python; no numpy needed.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence, TypeVar

from pydantic import BaseModel, Field, model_validator

from ai_eval.models import CaseOutcome

T = TypeVar("T")

DEFAULT_RESAMPLES = 2000
DEFAULT_LEVEL = 0.95
DEFAULT_SEED = 0


class ConfidenceInterval(BaseModel):
    """A point estimate with a two-sided ``level`` confidence interval over ``n`` units (cases)."""

    estimate: float
    low: float
    high: float
    level: float = Field(default=DEFAULT_LEVEL, gt=0, lt=1)
    n: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> ConfidenceInterval:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high})")
        return self

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def format(self, spec: str = ".3f") -> str:
        """e.g. ``0.800 [0.640, 0.920]``."""
        return f"{self.estimate:{spec}} [{self.low:{spec}}, {self.high:{spec}}]"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile (``q`` in [0, 1]) of already-sorted values."""
    if not sorted_values:
        raise ValueError("quantile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def bootstrap_ci(
    units: Sequence[T],
    statistic: Callable[[Sequence[T]], float] | None = None,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int | None = DEFAULT_SEED,
) -> ConfidenceInterval:
    """Percentile-bootstrap CI for ``statistic`` (default: the mean) over ``units``.

    ``units`` are the independent sampling units (one per case); each resample draws
    ``len(units)`` of them with replacement and applies ``statistic``. The point estimate is
    ``statistic(units)``. With ``seed`` set the result is deterministic. Raises ``ValueError``
    on empty input.
    """
    if not units:
        raise ValueError("bootstrap_ci needs at least one unit")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    stat = statistic or _mean  # type: ignore[assignment]
    estimate = float(stat(units))
    draws = _bootstrap_draws(units, stat, n_resamples, seed)
    return _percentile_ci(draws, estimate, level, len(units))


def _bootstrap_draws(
    units: Sequence[T], stat: Callable[[Sequence[T]], float], n_resamples: int, seed: int | None
) -> list[float]:
    """``n_resamples`` sorted values of ``stat`` over resamples of ``units`` (with replacement)."""
    n = len(units)
    rng = random.Random(seed)
    return sorted(float(stat([units[rng.randrange(n)] for _ in range(n)])) for _ in range(n_resamples))


def _percentile_ci(draws: Sequence[float], estimate: float, level: float, n: int) -> ConfidenceInterval:
    alpha = (1.0 - level) / 2.0
    low, high = quantile(draws, alpha), quantile(draws, 1.0 - alpha)
    # Guard float noise so the interval always brackets its own estimate on degenerate data.
    low, high = min(low, estimate), max(high, estimate)
    return ConfidenceInterval(estimate=estimate, low=low, high=high, level=level, n=n)


def _empty(level: float) -> ConfidenceInterval:
    return ConfidenceInterval(estimate=0.0, low=0.0, high=0.0, level=level, n=0)


def pass_rate_ci(
    outcomes: Sequence[CaseOutcome],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int | None = DEFAULT_SEED,
) -> ConfidenceInterval:
    """CI for the run pass rate (fraction of cases that passed, as in ``EvalRunSummary``).
    An empty run gives a zero-width interval at 0.0 with ``n == 0``."""
    if not outcomes:
        return _empty(level)
    flags = [1.0 if o.passed else 0.0 for o in outcomes]
    return bootstrap_ci(flags, n_resamples=n_resamples, level=level, seed=seed)


def mean_score_ci(
    outcomes: Sequence[CaseOutcome],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int | None = DEFAULT_SEED,
) -> ConfidenceInterval:
    """CI for the run mean score, matching ``EvalRunSummary.mean_score``: the mean of every
    scorer score of every sample. Cases are resampled whole (all their samples and scores), and
    each resample takes the pooled mean of its scores. When no case has any scores the summary
    falls back to the pass rate, and so does this. An empty run gives a zero-width interval at 0.0.
    """
    if not outcomes:
        return _empty(level)
    units = [
        (sum(s.score for smp in o.sample_list() for s in smp.scores),
         sum(len(smp.scores) for smp in o.sample_list()))
        for o in outcomes
    ]
    if not any(count for _, count in units):
        return pass_rate_ci(outcomes, n_resamples=n_resamples, level=level, seed=seed)

    def pooled(sample: Sequence[tuple[float, int]]) -> float:
        total = sum(c for _, c in sample)
        # A resample can draw only score-less cases (e.g. adapter errors); count that as 0.
        return sum(s for s, _ in sample) / total if total else 0.0

    return bootstrap_ci(units, pooled, n_resamples=n_resamples, level=level, seed=seed)


# --- paired significance for `compare` ---------------------------------------------------------
#
# Two runs of the same suite are *paired*: every case appears in both. Pairing removes the
# case-to-case variation (some cases are just hard) and leaves only the change itself, which is
# far more sensitive than comparing two independent pass rates.


class McNemarResult(BaseModel):
    """Exact McNemar test on paired pass/fail outcomes.

    Only discordant pairs carry information: ``regressions`` (baseline passed, candidate failed)
    and ``improvements`` (failed -> passed). Under the null hypothesis of no change each discordant
    pair is equally likely to go either way, so the two-sided p-value is an exact binomial test of
    ``regressions`` out of ``regressions + improvements`` at p = 0.5. No flips gives p = 1.
    """

    n_pairs: int = Field(ge=0)
    regressions: int = Field(ge=0)
    improvements: int = Field(ge=0)
    p_value: float = Field(ge=0, le=1)

    @property
    def discordant(self) -> int:
        return self.regressions + self.improvements


class PairedDelta(ConfidenceInterval):
    """Mean paired delta (candidate - baseline) with a paired-bootstrap CI and two-sided p-value.

    The p-value is the bootstrap estimate of how often the resampled mean delta lands on the other
    side of zero (doubled, with a +1 correction so it is never exactly 0). Identical runs give an
    interval of [0, 0] and p = 1.
    """

    p_value: float = Field(ge=0, le=1)


class CompareSignificance(BaseModel):
    """Significance of the change between two runs, over the cases present in both."""

    n_pairs: int = Field(ge=0)
    level: float = Field(default=DEFAULT_LEVEL, gt=0, lt=1)
    mcnemar: McNemarResult
    pass_rate_delta: PairedDelta | None = None
    score_delta: PairedDelta | None = None


def mcnemar_test(regressions: int, improvements: int, n_pairs: int | None = None) -> McNemarResult:
    """Exact two-sided McNemar test from the counts of discordant pairs."""
    if regressions < 0 or improvements < 0:
        raise ValueError("discordant counts must be >= 0")
    m = regressions + improvements
    if n_pairs is None:
        n_pairs = m
    if n_pairs < m:
        raise ValueError(f"n_pairs ({n_pairs}) must be >= regressions + improvements ({m})")
    if m == 0:
        p = 1.0
    else:
        k = min(regressions, improvements)
        tail = sum(math.comb(m, i) for i in range(k + 1))
        # Exact integer arithmetic before the single division keeps this accurate for large m.
        p = min(1.0, 2 * tail / 2**m)
    return McNemarResult(n_pairs=n_pairs, regressions=regressions, improvements=improvements, p_value=p)


def paired_bootstrap(
    deltas: Sequence[float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int | None = DEFAULT_SEED,
) -> PairedDelta:
    """Paired bootstrap over per-case deltas (candidate - baseline): CI and p-value for the mean."""
    if not deltas:
        raise ValueError("paired_bootstrap needs at least one pair")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    draws = _bootstrap_draws(deltas, _mean, n_resamples, seed)
    ci = _percentile_ci(draws, _mean(deltas), level, len(deltas))
    at_or_below = sum(1 for d in draws if d <= 0.0)
    at_or_above = sum(1 for d in draws if d >= 0.0)
    p = min(1.0, 2.0 * (min(at_or_below, at_or_above) + 1) / (n_resamples + 1))
    return PairedDelta(**ci.model_dump(), p_value=p)


def case_mean_score(outcome: CaseOutcome) -> float:
    """Mean of every scorer score across all of a case's samples; falls back to the case's sample
    pass rate when nothing was scored (e.g. adapter errors), as the run summary does."""
    scores = [s.score for smp in outcome.sample_list() for s in smp.scores]
    return sum(scores) / len(scores) if scores else outcome.pass_rate


def compare_significance(
    baseline: Sequence[CaseOutcome],
    candidate: Sequence[CaseOutcome],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int | None = DEFAULT_SEED,
) -> CompareSignificance:
    """Paired significance between two runs, matched by case id (unmatched cases are ignored).

    - ``mcnemar``: exact McNemar test on case pass/fail flips.
    - ``pass_rate_delta``: paired bootstrap of the per-case pass delta (-1, 0 or +1).
    - ``score_delta``: paired bootstrap of the per-case mean-score delta.

    Both deltas are ``None`` when the runs share no cases.
    """
    b_map = {o.case_id: o for o in baseline}
    c_map = {o.case_id: o for o in candidate}
    ids = sorted(set(b_map) & set(c_map))
    regressions = sum(1 for i in ids if b_map[i].passed and not c_map[i].passed)
    improvements = sum(1 for i in ids if not b_map[i].passed and c_map[i].passed)
    result = CompareSignificance(
        n_pairs=len(ids), level=level, mcnemar=mcnemar_test(regressions, improvements, len(ids))
    )
    if not ids:
        return result
    kw = {"n_resamples": n_resamples, "level": level, "seed": seed}
    result.pass_rate_delta = paired_bootstrap(
        [float(c_map[i].passed) - float(b_map[i].passed) for i in ids], **kw
    )
    result.score_delta = paired_bootstrap(
        [case_mean_score(c_map[i]) - case_mean_score(b_map[i]) for i in ids], **kw
    )
    return result
