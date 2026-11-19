from __future__ import annotations

import re
from typing import Any

from ai_eval.models import EvalCase, RunResult, ScoreResult
from ai_eval.scorers.base import Scorer
from ai_eval.scorers.trajectory import TrajectoryScorer, _as_call_list

# Tool errors that mean "the arguments were wrong" rather than "the tool failed" (e.g. an unknown
# SKU). Covers the agent loop's own argument errors, pydantic/MCP server validation errors and
# Python signature errors surfaced by tool servers.
ARG_ERROR_PATTERNS = (
    r"invalid json arguments",
    r"arguments must be an object",
    r"validation error",
    r"field required",
    r"missing (?:\d+ )?required",
    r"unexpected keyword argument",
    r"invalid (?:argument|parameter|input)s?\b",
    r"input should be",
)
_ARG_ERROR_RE = re.compile("|".join(ARG_ERROR_PATTERNS), re.IGNORECASE)

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}


def _type_ok(value: Any, schema: dict[str, Any]) -> bool:
    """Best-effort check of one value against a (sub)schema: ``type``, ``enum`` and ``anyOf``/
    ``oneOf`` of those. Unknown keywords are ignored, so this never rejects a valid value."""
    if "enum" in schema and value not in schema["enum"]:
        return False
    alternatives = schema.get("anyOf") or schema.get("oneOf")
    if (
        isinstance(alternatives, list)
        and alternatives
        and not any(isinstance(alt, dict) and _type_ok(value, alt) for alt in alternatives)
    ):
        return False
    types = schema.get("type")
    if types is None:
        return True
    for t in types if isinstance(types, list) else [types]:
        py = _JSON_TYPES.get(t)
        if py is None:
            return True  # unknown type keyword: don't guess
        if isinstance(value, bool) and t in ("integer", "number"):
            continue  # JSON booleans are not numbers
        if isinstance(value, py):
            return True
    return False


def schema_arg_errors(args: dict[str, Any] | None, schema: dict[str, Any]) -> list[str]:
    """Problems with ``args`` under a tool's JSON input schema: missing required keys, keys
    rejected by ``additionalProperties: false``, and top-level type/enum mismatches."""
    args = args or {}
    props = schema.get("properties") or {}
    problems = [f"missing required argument {k!r}" for k in schema.get("required") or [] if k not in args]
    if schema.get("additionalProperties") is False:
        problems += [f"unexpected argument {k!r}" for k in args if k not in props]
    for k, v in args.items():
        sub = props.get(k)
        if isinstance(sub, dict) and not _type_ok(v, sub):
            problems.append(f"argument {k!r} has invalid value {v!r}")
    return problems


class ToolUseScorer(Scorer):
    """Aggregate tool-use quality for agent runs: tool precision/recall, invalid-argument rate and
    efficiency, each with its own pass threshold.

    Reads ``run.trace["tool_calls"]`` (as recorded by the agent loop: ``{name, args, error, ...}``)
    and ``case.metadata["expected_tools"]`` (tool names or ``{name, args}`` dicts).

    - **precision**: matched calls / actual calls (1.0 when no calls were made). Extra, repeated
      and unknown-tool calls lower it.
    - **recall**: matched calls / expected calls (1.0 when nothing was expected).
    - **invalid_arg_rate**: share of actual calls whose arguments were invalid: the recorded
      error looks like an argument/validation error (see ``ARG_ERROR_PATTERNS``), or — when
      ``tool_schemas`` (``{tool_name: json_schema}``) is configured — the args break that tool's
      schema (missing required keys, type/enum mismatches, disallowed extra keys).
    - **efficiency**: optimal calls / actual calls, capped at 1.0. Optimal is
      ``case.metadata["optimal_steps"]`` when set, else the number of expected calls. Making
      fewer calls than optimal is not rewarded here; recall already penalizes it.

    Matching is order-insensitive, one actual call per expected call. With ``match_args=True``
    an expected entry with ``args`` only matches a call whose args contain those pairs.

    The case passes when ``precision >= min_precision``, ``recall >= min_recall``,
    ``invalid_arg_rate <= max_invalid_arg_rate`` and ``efficiency >= min_efficiency``. The score
    is the mean of precision, recall, ``1 - invalid_arg_rate`` and efficiency.

    Cases without ``expected_tools`` or without ``tool_calls`` in the trace are skipped (pass).
    A run error (e.g. ``max_steps`` reached) fails the case but metrics are still reported.
    """

    name = "tool_use"

    def __init__(
        self,
        match_args: bool = False,
        min_precision: float = 0.0,
        min_recall: float = 1.0,
        max_invalid_arg_rate: float = 0.0,
        min_efficiency: float = 0.0,
        tool_schemas: dict[str, dict[str, Any]] | None = None,
        trace_key: str = "tool_calls",
        expected_key: str = "expected_tools",
        optimal_key: str = "optimal_steps",
    ):
        thresholds = {
            "min_precision": min_precision,
            "min_recall": min_recall,
            "max_invalid_arg_rate": max_invalid_arg_rate,
            "min_efficiency": min_efficiency,
        }
        for key, value in thresholds.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"tool_use {key} must be between 0 and 1, got {value!r}")
        if tool_schemas is not None and not (
            isinstance(tool_schemas, dict) and all(isinstance(s, dict) for s in tool_schemas.values())
        ):
            raise ValueError("tool_use tool_schemas must map tool names to JSON schema objects")
        self.match_args = bool(match_args)
        self.min_precision = float(min_precision)
        self.min_recall = float(min_recall)
        self.max_invalid_arg_rate = float(max_invalid_arg_rate)
        self.min_efficiency = float(min_efficiency)
        self.tool_schemas = tool_schemas or {}
        self.trace_key = trace_key
        self.expected_key = expected_key
        self.optimal_key = optimal_key
        # Reuse the trajectory scorer's arg-aware, maximum bipartite matching.
        self._matcher = TrajectoryScorer(mode="any_order", match_args=self.match_args)

    def _skip(self, reason: str) -> ScoreResult:
        return ScoreResult(scorer=self.name, passed=True, score=1.0, details={"skipped": reason})

    def _invalid_reason(self, raw: Any, call: dict[str, Any]) -> str | None:
        error = raw.get("error") if isinstance(raw, dict) else None
        if isinstance(error, str) and _ARG_ERROR_RE.search(error):
            return error.strip().splitlines()[0][:200]
        schema = self.tool_schemas.get(call["name"])
        if schema is not None:
            problems = schema_arg_errors(call["args"], schema)
            if problems:
                return "; ".join(problems)
        return None

    def _optimal(self, case: EvalCase, expected: list[dict[str, Any]]) -> int:
        value = case.metadata.get(self.optimal_key)
        if value is None:
            return len(expected)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"{self.optimal_key} must be a non-negative integer, got {value!r}")
        return value

    def score(self, case: EvalCase, run: RunResult) -> ScoreResult:
        raw_calls = run.trace.get(self.trace_key)
        try:
            expected = _as_call_list(case.metadata.get(self.expected_key))
            actual = _as_call_list(raw_calls)
            optimal = self._optimal(case, expected or [])
        except TypeError as e:
            return ScoreResult(
                scorer=self.name, passed=False, score=0.0, details={"error": f"invalid tool calls: {e}"}
            )
        if expected is None:
            if run.error:
                return ScoreResult(scorer=self.name, passed=False, score=0.0, details={"error": run.error})
            return self._skip(f"no {self.expected_key} in case metadata")
        if actual is None:
            if run.error:
                return ScoreResult(scorer=self.name, passed=False, score=0.0, details={"error": run.error})
            return self._skip(f"no {self.trace_key} in run trace")

        pairs = self._matcher._match_unordered(expected, actual)
        matched = len(pairs)
        precision = matched / len(actual) if actual else 1.0
        recall = matched / len(expected) if expected else 1.0

        invalid: list[dict[str, Any]] = []
        for raw, call in zip(raw_calls, actual):
            reason = self._invalid_reason(raw, call)
            if reason is not None:
                invalid.append({"name": call["name"], "args": call["args"], "reason": reason})
        invalid_rate = len(invalid) / len(actual) if actual else 0.0

        if actual:
            efficiency = min(1.0, optimal / len(actual))
        else:
            efficiency = 1.0

        checks = {
            "precision": precision >= self.min_precision,
            "recall": recall >= self.min_recall,
            "invalid_arg_rate": invalid_rate <= self.max_invalid_arg_rate,
            "efficiency": efficiency >= self.min_efficiency,
        }
        value = (precision + recall + (1.0 - invalid_rate) + efficiency) / 4
        matched_exp = {i for i, _ in pairs}
        matched_act = {j for _, j in pairs}
        details: dict[str, Any] = {
            "precision": precision,
            "recall": recall,
            "invalid_arg_rate": invalid_rate,
            "efficiency": efficiency,
            "thresholds": {
                "min_precision": self.min_precision,
                "min_recall": self.min_recall,
                "max_invalid_arg_rate": self.max_invalid_arg_rate,
                "min_efficiency": self.min_efficiency,
            },
            "failed_checks": [k for k, ok in checks.items() if not ok],
            "match_args": self.match_args,
            "calls": len(actual),
            "expected_calls": len(expected),
            "optimal_calls": optimal,
            "matched": matched,
            "missing": [expected[i] for i in range(len(expected)) if i not in matched_exp],
            "extra": [actual[j] for j in range(len(actual)) if j not in matched_act],
            "invalid_calls": invalid,
        }
        if run.error:
            details["error"] = run.error
        return ScoreResult(
            scorer=self.name,
            passed=all(checks.values()) and not run.error,
            score=value,
            details=details,
        )

