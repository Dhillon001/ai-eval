from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
