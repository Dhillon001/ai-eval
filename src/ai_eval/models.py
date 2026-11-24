from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class HumanLabel(BaseModel):
    """A human verdict on a case, used to calibrate the LLM judge (``ai-eval calibrate``).

    ``passed`` is required (``pass`` is accepted as an alias); ``score`` is an optional 0–1 grade
    for correlating with judge scores; ``notes`` is free text for the labeler's reasoning.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    passed: bool = Field(validation_alias=AliasChoices("passed", "pass"))
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str | None = None


class EvalCase(BaseModel):
    """One test case, loaded from a JSONL line.

    ``metadata`` is free-form; keys read by built-in components:

    - ``tags``: list of strings, used for per-tag pass rates in ``compare``.
    - ``judge_criteria``: grading rubric for the ``llm_judge`` scorer.
    - ``expected_doc_ids``: relevant doc ids for the ``retrieval`` scorer.
    - ``expected_tools``: expected agent tool calls (tool names or ``{name, args}``) for the
      ``trajectory`` and ``tool_use`` scorers.
    - ``optimal_steps``: the minimum number of tool calls needed, for ``tool_use`` efficiency
      (defaults to the number of ``expected_tools``).
    - ``mock_retrieved_ids`` / ``mock_tool_calls``: emitted into the trace by the ``mock``
      adapter so retrieval and agent suites run offline.

    ``human_label`` is an optional human verdict for judge calibration; it can also be supplied
    from a separate labels JSONL (see ``ai_eval.dataset.load_labels``).
    """

    id: str
    input: str
    context: str | None = None
    expected: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    checks: list[str] = Field(default_factory=list)
    human_label: HumanLabel | None = None


class RunResult(BaseModel):
    case_id: str
    output: str
    trace: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


class ScoreResult(BaseModel):
    scorer: str
    passed: bool
    score: float
    details: dict[str, Any] = Field(default_factory=dict)


def _sample_passed(run: RunResult, scores: list[ScoreResult]) -> bool:
    if not scores:
        return run.error is None
    return all(s.passed for s in scores)


class CaseSample(BaseModel):
    """One of several runs of the same case (``ai-eval run --repeats N``)."""

    run: RunResult
    scores: list[ScoreResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return _sample_passed(self.run, self.scores)


class CaseOutcome(BaseModel):
    """The result of one case.

    ``run``/``scores`` hold the first sample. With ``--repeats N`` (N > 1), ``samples`` holds all N
    samples (the first one included) and the case passes when a strict majority of its samples
    pass; ``pass_rate`` and ``consistency`` describe the spread. Without repeats ``samples`` is
    empty and the outcome behaves exactly as a single run.
    """

    case_id: str
    run: RunResult
    scores: list[ScoreResult] = Field(default_factory=list)
    samples: list[CaseSample] = Field(default_factory=list)

    def sample_list(self) -> list[CaseSample]:
        """Every sample of this case: ``samples`` when repeated, else the single run."""
        return self.samples or [CaseSample(run=self.run, scores=self.scores)]

    @property
    def repeats(self) -> int:
        return len(self.samples) or 1

    @property
    def passed_samples(self) -> int:
        return sum(1 for s in self.sample_list() if s.passed)

    @property
    def pass_rate(self) -> float:
        """Fraction of samples that passed (0.0 or 1.0 for a single run)."""
        return self.passed_samples / self.repeats

    @property
    def consistency(self) -> float:
        """Fraction of samples that agree with the majority outcome: 1.0 when every sample
        passed or every sample failed, down to 0.5 for an even split."""
        k, n = self.passed_samples, self.repeats
        return max(k, n - k) / n

    @property
    def flaky(self) -> bool:
        """True when repeated samples disagree: at least one passed and at least one failed.
        A single run is never flaky."""
        return self.repeats > 1 and 0 < self.passed_samples < self.repeats

    @property
    def passed(self) -> bool:
        if self.samples:
            return self.passed_samples * 2 > len(self.samples)
        return _sample_passed(self.run, self.scores)


class SuiteConfig(BaseModel):
    name: str
    dataset: str
    scorers: list[dict[str, Any]] = Field(default_factory=list)
    adapter: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    # Per-model price overrides (USD per million tokens), merged over ai_eval.pricing.DEFAULT_PRICES.
    pricing: dict[str, dict[str, Any]] = Field(default_factory=dict)


class EvalRunSummary(BaseModel):
    run_id: str
    suite_name: str
    created_at: datetime
    adapter: str
    model: str | None
    config_hash: str
    git_sha: str | None
    total: int
    passed: int
    pass_rate: float
    mean_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


RUN_EXPORT_FORMAT_VERSION = 1


class RunExport(BaseModel):
    """A self-contained run (summary, outcomes, cases) that can be written to JSON and compared
    on another machine without access to the original runs.db."""

    format_version: int = RUN_EXPORT_FORMAT_VERSION
    summary: EvalRunSummary
    outcomes: list[CaseOutcome] = Field(default_factory=list)
    cases: list[EvalCase] = Field(default_factory=list)

    def cases_by_id(self) -> dict[str, EvalCase]:
        return {c.id: c for c in self.cases}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
