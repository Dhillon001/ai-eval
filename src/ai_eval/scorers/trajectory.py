from __future__ import annotations

import json
from typing import Any

from ai_eval.models import EvalCase, RunResult, ScoreResult
from ai_eval.scorers.base import Scorer

MODES = ("exact_order", "subset", "any_order")


def _normalize_call(item: Any) -> dict[str, Any]:
    """Coerce a tool call into ``{"name": str, "args": dict | None}``.

    Accepts a bare tool name string, ``{"name", "args"}``, ``{"name", "arguments"}``
    (arguments may be a JSON-encoded string), or OpenAI-style ``{"function": {...}}``.
    ``args`` is ``None`` when the entry does not specify any.
    """
    if isinstance(item, str):
        return {"name": item, "args": None}
    if not isinstance(item, dict):
        raise TypeError(f"expected a tool call dict or name, got {type(item).__name__}")
    if "name" not in item and isinstance(item.get("function"), dict):
        item = item["function"]
    name = item.get("name")
    if not isinstance(name, str) or not name:
        raise TypeError(f"tool call is missing a name: {item!r}")
    args = item.get("args", item.get("arguments"))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as e:
            raise TypeError(f"tool call {name!r} has non-JSON arguments: {e}") from None
    if args is not None and not isinstance(args, dict):
        raise TypeError(f"tool call {name!r} args must be an object, got {type(args).__name__}")
    return {"name": name, "args": args}


def _as_call_list(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"expected a list of tool calls, got {type(value).__name__}")
    return [_normalize_call(v) for v in value]


class TrajectoryScorer(Scorer):
    """Score an agent's tool-use trajectory by comparing ``run.trace["tool_calls"]``
    (a list of ``{name, args}``) with ``case.metadata["expected_tools"]``.

    Expected entries may be bare tool names or ``{name, args}`` dicts. With
    ``match_args=True``, an expected entry that lists ``args`` only matches an actual call
    whose args contain those key/value pairs (extra actual args are allowed); entries
    without ``args`` match on name alone.

    Modes (score is always the fraction of calls matched, in [0, 1]):

    - ``exact_order``: the trajectory must be exactly the expected sequence. Matching is
      order-preserving (longest common subsequence); score = matched / max(len(expected),
      len(actual)), so extra, missing and out-of-order calls all lower it.
    - ``any_order``: same calls as expected, order ignored, no extras. Score = matched /
      max(len(expected), len(actual)).
    - ``subset``: every expected call must appear somewhere; order ignored and extra calls
      are allowed. Score = matched / len(expected).

    Cases without ``expected_tools`` in metadata or ``tool_calls`` in the trace are skipped
    (pass). An explicit empty ``expected_tools`` list means "no tool calls expected".
    """

    name = "trajectory"

    def __init__(
        self,
        mode: str = "exact_order",
        match_args: bool = False,
        threshold: float = 1.0,
        trace_key: str = "tool_calls",
        expected_key: str = "expected_tools",
    ):
        mode = mode.strip().lower().replace("-", "_")
        if mode not in MODES:
            raise ValueError(f"Unknown trajectory mode {mode!r}; expected one of {', '.join(MODES)}")
        self.mode = mode
        self.match_args = bool(match_args)
        self.threshold = float(threshold)
        self.trace_key = trace_key
        self.expected_key = expected_key
        self.name = f"trajectory:{mode}"

    def _skip(self, reason: str) -> ScoreResult:
        return ScoreResult(scorer=self.name, passed=True, score=1.0, details={"skipped": reason})

    def _matches(self, expected: dict[str, Any], actual: dict[str, Any]) -> bool:
        if expected["name"] != actual["name"]:
            return False
        if not self.match_args or expected["args"] is None:
            return True
        actual_args = actual["args"] or {}
        return all(k in actual_args and actual_args[k] == v for k, v in expected["args"].items())

    def _match_ordered(self, expected: list[dict], actual: list[dict]) -> list[tuple[int, int]]:
        """Longest common subsequence under ``_matches``; returns (expected_idx, actual_idx)."""
        n, m = len(expected), len(actual)
        table = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if self._matches(expected[i], actual[j]):
                    table[i][j] = table[i + 1][j + 1] + 1
                else:
                    table[i][j] = max(table[i + 1][j], table[i][j + 1])
        pairs: list[tuple[int, int]] = []
        i = j = 0
        while i < n and j < m:
            if self._matches(expected[i], actual[j]) and table[i][j] == table[i + 1][j + 1] + 1:
                pairs.append((i, j))
                i += 1
                j += 1
            elif table[i + 1][j] >= table[i][j + 1]:
                i += 1
            else:
                j += 1
        return pairs

    def _match_unordered(self, expected: list[dict], actual: list[dict]) -> list[tuple[int, int]]:
        """Maximum bipartite matching, so a name-only expectation never steals the one
        actual call a more specific (args-constrained) expectation needs."""
        owner: dict[int, int] = {}  # actual_idx -> expected_idx

        def assign(i: int, seen: set[int]) -> bool:
            for j, call in enumerate(actual):
                if j in seen or not self._matches(expected[i], call):
                    continue
                seen.add(j)
                if j not in owner or assign(owner[j], seen):
                    owner[j] = i
                    return True
            return False

        for i in range(len(expected)):
            assign(i, set())
        return sorted((i, j) for j, i in owner.items())

    def score(self, case: EvalCase, run: RunResult) -> ScoreResult:
        if run.error:
            return ScoreResult(scorer=self.name, passed=False, score=0.0, details={"error": run.error})

        try:
            expected = _as_call_list(case.metadata.get(self.expected_key))
            actual = _as_call_list(run.trace.get(self.trace_key))
        except TypeError as e:
            return ScoreResult(
                scorer=self.name,
                passed=False,
                score=0.0,
                details={"error": f"invalid tool calls: {e}"},
            )
        if expected is None:
            return self._skip(f"no {self.expected_key} in case metadata")
        if actual is None:
            return self._skip(f"no {self.trace_key} in run trace")

        if self.mode == "exact_order":
            pairs = self._match_ordered(expected, actual)
        else:
            pairs = self._match_unordered(expected, actual)

        if self.mode == "subset":
            denom = len(expected)
        else:
            denom = max(len(expected), len(actual))
        value = len(pairs) / denom if denom else 1.0

        matched_exp = {i for i, _ in pairs}
        matched_act = {j for _, j in pairs}
        missing = [expected[i] for i in range(len(expected)) if i not in matched_exp]
        extra = [actual[j] for j in range(len(actual)) if j not in matched_act]
        # Missing expectations whose tool *was* called, just with the wrong args.
        arg_mismatches = (
            [e["name"] for e in missing if any(a["name"] == e["name"] for a in actual)]
            if self.match_args
            else []
        )

        return ScoreResult(
            scorer=self.name,
            passed=value >= self.threshold,
            score=value,
            details={
                "mode": self.mode,
                "match_args": self.match_args,
                "threshold": self.threshold,
                "expected": expected,
                "actual": actual,
                "matched": len(pairs),
                "missing": missing,
                "extra": extra,  # allowed (not penalized) in subset mode
                "arg_mismatches": arg_mismatches,
            },
        )
