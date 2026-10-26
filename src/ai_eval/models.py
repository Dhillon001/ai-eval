from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """One test case, loaded from a JSONL line.

    ``metadata`` is free-form; keys read by built-in components:

    - ``tags``: list of strings, used for per-tag pass rates in ``compare``.
    - ``judge_criteria``: grading rubric for the ``llm_judge`` scorer.
    - ``expected_doc_ids``: relevant doc ids for the ``retrieval`` scorer.
    - ``expected_tools``: expected agent tool calls (tool names or ``{name, args}``) for the
      ``trajectory`` scorer.
    - ``mock_retrieved_ids`` / ``mock_tool_calls``: emitted into the trace by the ``mock``
      adapter so retrieval and agent suites run offline.
    """

    id: str
    input: str
    context: str | None = None
    expected: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    checks: list[str] = Field(default_factory=list)


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


class CaseOutcome(BaseModel):
    case_id: str
    run: RunResult
    scores: list[ScoreResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        if not self.scores:
            return self.run.error is None
        return all(s.passed for s in self.scores)


class SuiteConfig(BaseModel):
    name: str
    dataset: str
    scorers: list[dict[str, Any]] = Field(default_factory=list)
    adapter: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


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
