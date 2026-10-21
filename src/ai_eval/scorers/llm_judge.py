from __future__ import annotations

import json
import os
import re

from ai_eval.models import EvalCase, RunResult, ScoreResult
from ai_eval.scorers.base import Scorer

DEFAULT_RUBRIC = """Rate whether the assistant response satisfies the criteria.
Return JSON only: {"pass": true|false, "score": 0.0-1.0, "reason": "..."}"""


class LLMJudgeScorer(Scorer):
    name = "llm_judge"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        rubric: str | None = None,
        pass_threshold: float = 0.7,
    ):
        self.model = model
        self.rubric = rubric or DEFAULT_RUBRIC
        self.pass_threshold = pass_threshold

    def score(self, case: EvalCase, run: RunResult) -> ScoreResult:
        if run.error:
            return ScoreResult(
                scorer=self.name,
                passed=False,
                score=0.0,
                details={"error": run.error},
            )

        try:
            from openai import OpenAI
        except ImportError:
            return ScoreResult(
                scorer=self.name,
                passed=False,
                score=0.0,
                details={"error": "openai package not installed"},
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return ScoreResult(
                scorer=self.name,
                passed=False,
                score=0.0,
                details={"error": "OPENAI_API_KEY is not set"},
            )

        client = OpenAI(api_key=api_key)
        criteria = case.metadata.get("judge_criteria") or case.expected or "Response is helpful and accurate."
        prompt = f"""{self.rubric}

User input:
{case.input}

Assistant response:
{run.output}

Criteria:
{criteria}
"""
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an evaluation judge. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or "{}"
        except Exception as e:
            return ScoreResult(
                scorer=self.name,
                passed=False,
                score=0.0,
                details={"error": str(e)},
            )

        parsed = self._parse_json(raw)
        score = float(parsed.get("score", 0.0))
        passed = bool(parsed.get("pass", score >= self.pass_threshold))
        if "pass" not in parsed:
            passed = score >= self.pass_threshold

        return ScoreResult(
            scorer=self.name,
            passed=passed,
            score=max(0.0, min(1.0, score)),
            details={"raw": raw, "parsed": parsed},
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {"pass": False, "score": 0.0, "reason": "failed to parse judge output"}
