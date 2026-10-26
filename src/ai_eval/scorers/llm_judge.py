from __future__ import annotations

import json
import re
from typing import Any

from ai_eval.cache import CachedResponse, JudgeCache, cache_key
from ai_eval.llm import DEFAULT_MAX_RETRIES, LLMError, chat, make_client
from ai_eval.models import EvalCase, RunResult, ScoreResult
from ai_eval.scorers.base import Scorer

DEFAULT_RUBRIC = """Rate whether the assistant response satisfies the criteria.
Return JSON only: {"pass": true|false, "score": 0.0-1.0, "reason": "..."}"""

FAITHFULNESS_RUBRIC = """You are checking a retrieval-augmented answer for faithfulness.
A response is faithful only if every factual claim it makes is supported by the provided context.
Unsupported or contradicted claims make it unfaithful, even if they are true in general.
If the context does not contain the answer, the faithful response says so instead of guessing.
Return JSON only: {"pass": true|false, "score": 0.0-1.0, "reason": "...", "unsupported_claims": ["..."]}"""

DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-5-5"}

JUDGE_SYSTEM = "You are an evaluation judge. Output JSON only."


class LLMJudgeScorer(Scorer):
    """LLM-as-judge. `mode: faithfulness` grades the answer against `case.context` (RAG)."""

    name = "llm_judge"

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        rubric: str | None = None,
        pass_threshold: float = 0.7,
        mode: str = "criteria",
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: Any = None,
        cache: JudgeCache | None = None,
    ):
        if mode not in ("criteria", "faithfulness"):
            raise ValueError("llm_judge mode must be 'criteria' or 'faithfulness'")
        self.provider = provider
        self.model = model or DEFAULT_MODELS.get(provider, "gpt-4o-mini")
        self.mode = mode
        self.rubric = rubric or (FAITHFULNESS_RUBRIC if mode == "faithfulness" else DEFAULT_RUBRIC)
        self.pass_threshold = pass_threshold
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self._client = client
        self.cache = cache
        if mode == "faithfulness":
            self.name = "faithfulness"

    def _fail(self, msg: str) -> ScoreResult:
        return ScoreResult(scorer=self.name, passed=False, score=0.0, details={"error": msg})

    def _build_prompt(self, case: EvalCase, run: RunResult) -> str | None:
        if self.mode == "faithfulness":
            if not case.context:
                return None
            return f"""{self.rubric}

Context:
{case.context}

User question:
{case.input}

Assistant response:
{run.output}
"""
        criteria = case.metadata.get("judge_criteria") or case.expected or "Response is helpful and accurate."
        return f"""{self.rubric}

User input:
{case.input}

Assistant response:
{run.output}

Criteria:
{criteria}
"""

    def score(self, case: EvalCase, run: RunResult) -> ScoreResult:
        if run.error:
            return self._fail(run.error)

        prompt = self._build_prompt(case, run)
        if prompt is None:
            return ScoreResult(scorer=self.name, passed=True, score=1.0, details={"skipped": "no context"})

        key = cache_key(self.provider, self.model, prompt, system=JUDGE_SYSTEM)
        cached = self.cache.get(key) if self.cache is not None else None
        if cached is not None:
            resp = cached
        else:
            try:
                if self._client is None:
                    self._client = make_client(self.provider)
                resp = chat(
                    self._client,
                    self.provider,
                    self.model,
                    JUDGE_SYSTEM,
                    prompt,
                    temperature=0.0,
                    max_retries=self.max_retries,
                )
            except LLMError as e:
                return self._fail(str(e))
            except Exception as e:
                return self._fail(f"{type(e).__name__}: {e}")
            if self.cache is not None:
                self.cache.put(
                    key,
                    CachedResponse(
                        text=resp.text or "",
                        prompt_tokens=resp.prompt_tokens,
                        completion_tokens=resp.completion_tokens,
                    ),
                )

        raw = resp.text or "{}"
        parsed = self._parse_json(raw)
        try:
            score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        except (TypeError, ValueError):
            score = 0.0
        if "pass" in parsed:
            passed = bool(parsed["pass"])
        else:
            passed = score >= self.pass_threshold

        return ScoreResult(
            scorer=self.name,
            passed=passed,
            score=score,
            details={
                "raw": raw,
                "parsed": parsed,
                "judge": f"{self.provider}:{self.model}",
                "cached": cached is not None,
            },
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidates = [fenced.group(1)] if fenced else []
        candidates.append(text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidates.append(match.group())
        for c in candidates:
            try:
                data = json.loads(c)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
        return {"pass": False, "score": 0.0, "reason": "failed to parse judge output"}
