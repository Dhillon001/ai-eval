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
    n = len(units)
    rng = random.Random(seed)
    draws = sorted(float(stat([units[rng.randrange(n)] for _ in range(n)])) for _ in range(n_resamples))
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
