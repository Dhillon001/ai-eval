from __future__ import annotations

import pytest

from ai_eval.models import EvalCase, RunResult
from ai_eval.scorers.trajectory import TrajectoryScorer


def _case(expected=None, **metadata) -> EvalCase:
    if expected is not None:
        metadata["expected_tools"] = expected
    return EvalCase(id="c1", input="q", metadata=metadata)


def _run(calls=None, error: str | None = None, **trace) -> RunResult:
    if calls is not None:
        trace["tool_calls"] = calls
    return RunResult(case_id="c1", output="a", trace=trace, error=error)


def _calls(*names: str) -> list[dict]:
    return [{"name": n, "args": {}} for n in names]


# --- construction -----------------------------------------------------------


def test_default_mode_and_name():
    scorer = TrajectoryScorer()
    assert scorer.mode == "exact_order"
    assert scorer.name == "trajectory:exact_order"


def test_mode_is_normalized():
    assert TrajectoryScorer(mode="Any-Order").mode == "any_order"


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown trajectory mode"):
        TrajectoryScorer(mode="fuzzy")


# --- exact_order ------------------------------------------------------------


def test_exact_order_perfect_match():
    s = TrajectoryScorer().score(_case(["search", "fetch"]), _run(_calls("search", "fetch")))
    assert s.passed
    assert s.score == 1.0
    assert s.details["missing"] == []
    assert s.details["extra"] == []
    assert s.scorer == "trajectory:exact_order"


def test_exact_order_wrong_order_partial():
    s = TrajectoryScorer().score(_case(["search", "fetch"]), _run(_calls("fetch", "search")))
    assert not s.passed
    assert s.score == pytest.approx(0.5)
    assert s.details["matched"] == 1


def test_exact_order_extra_call_penalized():
    s = TrajectoryScorer().score(
        _case(["search", "fetch"]), _run(_calls("search", "calc", "fetch"))
    )
    assert not s.passed
    assert s.score == pytest.approx(2 / 3)
    assert [c["name"] for c in s.details["extra"]] == ["calc"]


def test_exact_order_missing_call_penalized():
    s = TrajectoryScorer().score(_case(["search", "fetch", "answer"]), _run(_calls("search")))
    assert not s.passed
    assert s.score == pytest.approx(1 / 3)
    assert [c["name"] for c in s.details["missing"]] == ["fetch", "answer"]


def test_exact_order_threshold():
    scorer = TrajectoryScorer(threshold=0.5)
    s = scorer.score(_case(["search", "fetch"]), _run(_calls("fetch", "search")))
    assert s.passed
    assert s.details["threshold"] == 0.5


# --- any_order --------------------------------------------------------------


def test_any_order_ignores_order():
    scorer = TrajectoryScorer(mode="any_order")
    s = scorer.score(_case(["search", "fetch"]), _run(_calls("fetch", "search")))
    assert s.passed
    assert s.score == 1.0


def test_any_order_extra_call_penalized():
    scorer = TrajectoryScorer(mode="any_order")
    s = scorer.score(_case(["search", "fetch"]), _run(_calls("fetch", "calc", "search")))
    assert not s.passed
    assert s.score == pytest.approx(2 / 3)


def test_any_order_missing_call_penalized():
    scorer = TrajectoryScorer(mode="any_order")
    s = scorer.score(_case(["search", "fetch"]), _run(_calls("fetch")))
    assert not s.passed
    assert s.score == pytest.approx(0.5)


def test_any_order_duplicate_calls_count_once():
    scorer = TrajectoryScorer(mode="any_order")
    s = scorer.score(_case(["search"]), _run(_calls("search", "search")))
    assert s.score == pytest.approx(0.5)
    assert len(s.details["extra"]) == 1


# --- subset -----------------------------------------------------------------


def test_subset_allows_extra_calls():
    scorer = TrajectoryScorer(mode="subset")
    s = scorer.score(_case(["fetch", "search"]), _run(_calls("search", "calc", "fetch")))
    assert s.passed
    assert s.score == 1.0
    assert [c["name"] for c in s.details["extra"]] == ["calc"]


def test_subset_missing_call():
    scorer = TrajectoryScorer(mode="subset")
    s = scorer.score(_case(["search", "fetch"]), _run(_calls("search", "calc")))
    assert not s.passed
    assert s.score == pytest.approx(0.5)
    assert [c["name"] for c in s.details["missing"]] == ["fetch"]


# --- arg matching -----------------------------------------------------------


def test_args_ignored_without_match_args():
    expected = [{"name": "search", "args": {"q": "paris"}}]
    s = TrajectoryScorer().score(_case(expected), _run([{"name": "search", "args": {"q": "rome"}}]))
    assert s.passed
    assert s.details["arg_mismatches"] == []


def test_arg_mismatch_with_match_args():
    expected = [{"name": "search", "args": {"q": "paris"}}]
    scorer = TrajectoryScorer(match_args=True)
    s = scorer.score(_case(expected), _run([{"name": "search", "args": {"q": "rome"}}]))
    assert not s.passed
    assert s.score == 0.0
    assert s.details["arg_mismatches"] == ["search"]


def test_match_args_allows_extra_actual_args():
    expected = [{"name": "search", "args": {"q": "paris"}}]
    actual = [{"name": "search", "args": {"q": "paris", "limit": 5}}]
    s = TrajectoryScorer(match_args=True).score(_case(expected), _run(actual))
    assert s.passed


def test_match_args_name_only_expectation_matches_any_args():
    s = TrajectoryScorer(match_args=True).score(
        _case(["search"]), _run([{"name": "search", "args": {"q": "x"}}])
    )
    assert s.passed


def test_unordered_matching_prefers_specific_expectation():
    # A greedy matcher would let the name-only "search" take the q=paris call first.
    expected = ["search", {"name": "search", "args": {"q": "paris"}}]
    actual = [
        {"name": "search", "args": {"q": "paris"}},
        {"name": "search", "args": {"q": "rome"}},
    ]
    s = TrajectoryScorer(mode="any_order", match_args=True).score(_case(expected), _run(actual))
    assert s.passed
    assert s.score == 1.0


def test_json_string_arguments_and_openai_shape():
    expected = [{"name": "search", "args": {"q": "paris"}}]
    actual = [{"function": {"name": "search", "arguments": '{"q": "paris"}'}}]
    s = TrajectoryScorer(match_args=True).score(_case(expected), _run(actual))
    assert s.passed
    assert s.details["actual"] == [{"name": "search", "args": {"q": "paris"}}]


# --- skips and errors -------------------------------------------------------


def test_no_trace_skips():
    s = TrajectoryScorer().score(_case(["search"]), _run())
    assert s.passed
    assert "skipped" in s.details


def test_no_expected_tools_skips():
    s = TrajectoryScorer().score(_case(), _run(_calls("search")))
    assert s.passed
    assert "skipped" in s.details


def test_empty_expected_means_no_calls():
    scorer = TrajectoryScorer()
    assert scorer.score(_case([]), _run([])).passed
    s = scorer.score(_case([]), _run(_calls("search")))
    assert not s.passed
    assert s.score == 0.0


def test_empty_expected_subset_passes_with_calls():
    s = TrajectoryScorer(mode="subset").score(_case([]), _run(_calls("search")))
    assert s.passed


def test_run_error_fails():
    s = TrajectoryScorer().score(_case(["search"]), _run(_calls("search"), error="boom"))
    assert not s.passed
    assert s.details["error"] == "boom"


@pytest.mark.parametrize("bad", ["search", {"name": "search"}, [{"args": {}}], [42]])
def test_malformed_tool_calls_fail(bad):
    s = TrajectoryScorer().score(_case(["search"]), _run(bad))
    assert not s.passed
    assert "invalid tool calls" in s.details["error"]


def test_bad_json_arguments_fail():
    s = TrajectoryScorer().score(
        _case(["search"]), _run([{"name": "search", "arguments": "{not json"}])
    )
    assert not s.passed
    assert "non-JSON" in s.details["error"]
