from __future__ import annotations

from pathlib import Path

import pytest

from ai_eval.models import EvalCase, RunResult
from ai_eval.runner import run_suite
from ai_eval.scorers import build_scorer
from ai_eval.scorers.tool_use import ToolUseScorer, schema_arg_errors
from ai_eval.store import RunStore

ROOT = Path(__file__).resolve().parents[1]


def _case(expected=None, **metadata) -> EvalCase:
    if expected is not None:
        metadata["expected_tools"] = expected
    return EvalCase(id="c1", input="q", metadata=metadata)


def _run(calls=None, error: str | None = None, **trace) -> RunResult:
    if calls is not None:
        trace["tool_calls"] = calls
    return RunResult(case_id="c1", output="a", trace=trace, error=error)


def _call(name: str, error: str | None = None, **args) -> dict:
    return {"name": name, "args": args, "result": "" if error else "ok", "error": error}


SKU_SCHEMA = {
    "type": "object",
    "properties": {
        "sku": {"type": "string"},
        "quantity": {"type": "integer"},
        "warehouse": {"type": "string", "enum": ["SEA", "BOS"]},
    },
    "required": ["sku"],
    "additionalProperties": False,
}


# --- construction / wiring --------------------------------------------------


def test_defaults():
    s = ToolUseScorer()
    assert s.name == "tool_use"
    assert (s.min_precision, s.min_recall, s.max_invalid_arg_rate, s.min_efficiency) == (0.0, 1.0, 0.0, 0.0)
    assert s.match_args is False
    assert s.tool_schemas == {}


@pytest.mark.parametrize(
    "key", ["min_precision", "min_recall", "max_invalid_arg_rate", "min_efficiency"]
)
@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_thresholds_out_of_range_raise(key, value):
    with pytest.raises(ValueError, match=key):
        ToolUseScorer(**{key: value})


@pytest.mark.parametrize("schemas", [["not", "a", "dict"], {"get_part": "string"}])
def test_bad_tool_schemas_raise(schemas):
    with pytest.raises(ValueError, match="tool_schemas"):
        ToolUseScorer(tool_schemas=schemas)


def test_build_scorer_tool_use_with_options():
    scorer = build_scorer(
        {
            "type": "tool_use",
            "match_args": True,
            "min_precision": 0.5,
            "min_recall": 0.75,
            "max_invalid_arg_rate": 0.25,
            "min_efficiency": 0.5,
            "tool_schemas": {"get_part": SKU_SCHEMA},
        }
    )
    assert isinstance(scorer, ToolUseScorer)
    assert scorer.match_args is True
    assert (scorer.min_precision, scorer.min_recall, scorer.max_invalid_arg_rate, scorer.min_efficiency) == (
        0.5,
        0.75,
        0.25,
        0.5,
    )
    assert scorer.tool_schemas == {"get_part": SKU_SCHEMA}


def test_build_scorer_tool_use_rejects_bad_threshold():
    with pytest.raises(ValueError, match="min_recall"):
        build_scorer({"type": "tool_use", "min_recall": 2})


# --- precision / recall -----------------------------------------------------


def test_perfect_run():
    s = ToolUseScorer().score(
        _case(["check_stock", "reserve_part"]), _run([_call("check_stock"), _call("reserve_part")])
    )
    assert s.passed
    assert s.score == 1.0
    d = s.details
    assert (d["precision"], d["recall"], d["invalid_arg_rate"], d["efficiency"]) == (1.0, 1.0, 0.0, 1.0)
    assert (d["calls"], d["expected_calls"], d["optimal_calls"], d["matched"]) == (2, 2, 2, 2)
    assert d["missing"] == [] and d["extra"] == [] and d["invalid_calls"] == []
    assert d["failed_checks"] == []


def test_order_is_ignored():
    s = ToolUseScorer().score(_case(["a", "b"]), _run([_call("b"), _call("a")]))
    assert s.passed and s.details["precision"] == 1.0 and s.details["recall"] == 1.0


def test_extra_call_lowers_precision_and_efficiency_only():
    s = ToolUseScorer().score(_case(["check_stock"]), _run([_call("list_warehouses"), _call("check_stock")]))
    d = s.details
    assert d["precision"] == pytest.approx(0.5)
    assert d["recall"] == 1.0
    assert d["efficiency"] == pytest.approx(0.5)
    assert d["extra"] == [{"name": "list_warehouses", "args": {}}]
    assert s.score == pytest.approx((0.5 + 1.0 + 1.0 + 0.5) / 4)
    assert s.passed  # default thresholds only require full recall


def test_repeated_call_counts_once():
    s = ToolUseScorer().score(_case(["get_part"]), _run([_call("get_part"), _call("get_part"), _call("get_part")]))
    assert s.details["matched"] == 1
    assert s.details["precision"] == pytest.approx(1 / 3)
    assert len(s.details["extra"]) == 2


def test_missing_call_fails_default_recall():
    s = ToolUseScorer().score(_case(["check_stock", "reserve_part"]), _run([_call("check_stock")]))
    assert not s.passed
    assert s.details["recall"] == pytest.approx(0.5)
    assert s.details["precision"] == 1.0
    assert s.details["missing"] == [{"name": "reserve_part", "args": None}]
    assert s.details["failed_checks"] == ["recall"]


def test_unknown_tool_is_extra():
    s = ToolUseScorer().score(_case(["get_part"]), _run([_call("delete_everything")]))
    assert not s.passed
    assert (s.details["precision"], s.details["recall"]) == (0.0, 0.0)


def test_no_calls_expected_and_none_made():
    s = ToolUseScorer().score(_case([]), _run([]))
    assert s.passed and s.score == 1.0
    assert (s.details["precision"], s.details["recall"], s.details["efficiency"]) == (1.0, 1.0, 1.0)


def test_no_calls_expected_but_some_made():
    s = ToolUseScorer(min_precision=1.0).score(_case([]), _run([_call("get_part")]))
    assert s.details["precision"] == 0.0
    assert s.details["recall"] == 1.0
    assert s.details["efficiency"] == 0.0  # optimal is 0 calls
    assert not s.passed and s.details["failed_checks"] == ["precision"]


def test_calls_expected_but_none_made():
    s = ToolUseScorer().score(_case(["get_part"]), _run([]))
    assert not s.passed
    assert s.details["precision"] == 1.0  # nothing wrong was called
    assert s.details["recall"] == 0.0


# --- argument matching ------------------------------------------------------


def test_args_ignored_without_match_args():
    s = ToolUseScorer().score(
        _case([{"name": "get_part", "args": {"sku": "BRK-001"}}]), _run([_call("get_part", sku="FLT-204")])
    )
    assert s.passed and s.details["matched"] == 1


def test_match_args_requires_expected_pairs():
    case = _case([{"name": "get_part", "args": {"sku": "BRK-001"}}])
    wrong = ToolUseScorer(match_args=True).score(case, _run([_call("get_part", sku="FLT-204")]))
    assert not wrong.passed
    assert wrong.details["matched"] == 0
    right = ToolUseScorer(match_args=True).score(case, _run([_call("get_part", sku="BRK-001", verbose=True)]))
    assert right.passed  # extra actual args are allowed


def test_match_args_entry_without_args_matches_by_name():
    s = ToolUseScorer(match_args=True).score(
        _case([{"name": "search_parts"}]), _run([_call("search_parts", query="rotor")])
    )
    assert s.passed


def test_match_args_uses_best_assignment():
    # Greedy matching would pair the bare "check_stock" with the BOS call and miss the SEA one.
    case = _case(["check_stock", {"name": "check_stock", "args": {"warehouse": "BOS"}}])
    run = _run([_call("check_stock", warehouse="BOS"), _call("check_stock", warehouse="SEA")])
    s = ToolUseScorer(match_args=True).score(case, run)
    assert s.details["matched"] == 2 and s.passed


# --- invalid arguments ------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        "invalid JSON arguments: Expecting value: line 1 column 1 (char 0)",
        "arguments must be an object, got list",
        "1 validation error for get_partArguments\nsku\n  Field required [type=missing]",
        "TypeError: get_part() missing 1 required positional argument: 'sku'",
        "TypeError: check_stock() got an unexpected keyword argument 'location'",
        "Invalid argument: quantity must be positive",
        "Input should be a valid integer",
    ],
)
def test_argument_errors_count_as_invalid(error):
    s = ToolUseScorer().score(_case(["get_part"]), _run([_call("get_part", error=error)]))
    assert s.details["invalid_arg_rate"] == 1.0
    assert s.details["invalid_calls"][0]["name"] == "get_part"
    assert s.details["invalid_calls"][0]["reason"] == error.splitlines()[0]
    assert not s.passed and s.details["failed_checks"] == ["invalid_arg_rate"]


@pytest.mark.parametrize(
    "error",
    ["Unknown SKU: XYZ-999", "TimeoutError: tool call timed out after 15s", "tool reported an error"],
)
def test_tool_failures_are_not_argument_errors(error):
    s = ToolUseScorer().score(_case(["get_part"]), _run([_call("get_part", error=error)]))
    assert s.details["invalid_arg_rate"] == 0.0
    assert s.passed


def test_invalid_reason_is_truncated_to_one_line():
    error = "validation error " + "x" * 500 + "\nsecond line"
    s = ToolUseScorer().score(_case(["t"]), _run([_call("t", error=error)]))
    reason = s.details["invalid_calls"][0]["reason"]
    assert len(reason) == 200 and "\n" not in reason


def test_invalid_rate_is_share_of_calls_and_threshold_applies():
    calls = [
        _call("check_stock", error="invalid arguments: sku"),
        _call("check_stock", sku="BRK-001"),
        _call("reserve_part", sku="BRK-001", quantity=1),
        _call("get_part", sku="BRK-001"),
    ]
    case = _case(["check_stock", "reserve_part", "get_part"])
    assert ToolUseScorer(max_invalid_arg_rate=0.25).score(case, _run(calls)).passed
    s = ToolUseScorer(max_invalid_arg_rate=0.2).score(case, _run(calls))
    assert s.details["invalid_arg_rate"] == pytest.approx(0.25)
    assert not s.passed and s.details["failed_checks"] == ["invalid_arg_rate"]


def test_tool_schema_flags_bad_args():
    scorer = ToolUseScorer(tool_schemas={"reserve_part": SKU_SCHEMA}, max_invalid_arg_rate=1.0)
    calls = [
        _call("reserve_part", quantity=2),  # missing sku
        _call("reserve_part", sku="BRK-001", qty=2),  # unexpected key
        _call("reserve_part", sku="BRK-001", quantity="two"),  # wrong type
        _call("reserve_part", sku="BRK-001", warehouse="LAX"),  # not in enum
        _call("reserve_part", sku="BRK-001", quantity=2, warehouse="SEA"),  # fine
        _call("get_part", anything=1),  # no schema for this tool
    ]
    s = scorer.score(_case(["reserve_part"]), _run(calls))
    reasons = [c["reason"] for c in s.details["invalid_calls"]]
    assert len(reasons) == 4
    assert "missing required argument 'sku'" in reasons[0]
    assert "unexpected argument 'qty'" in reasons[1]
    assert "argument 'quantity' has invalid value 'two'" in reasons[2]
    assert "argument 'warehouse' has invalid value 'LAX'" in reasons[3]
    assert s.details["invalid_arg_rate"] == pytest.approx(4 / 6)


def test_recorded_error_takes_precedence_over_schema():
    scorer = ToolUseScorer(tool_schemas={"get_part": SKU_SCHEMA})
    s = scorer.score(_case(["get_part"]), _run([_call("get_part", error="validation error: sku")]))
    assert s.details["invalid_calls"][0]["reason"] == "validation error: sku"


def test_schema_arg_errors_helpers():
    assert schema_arg_errors({"sku": "A"}, SKU_SCHEMA) == []
    assert schema_arg_errors(None, SKU_SCHEMA) == ["missing required argument 'sku'"]
    # Booleans are not integers in JSON schema.
    assert schema_arg_errors({"sku": "A", "quantity": True}, SKU_SCHEMA) == [
        "argument 'quantity' has invalid value True"
    ]
    # Extra keys are fine unless additionalProperties is false.
    open_schema = {"properties": {"n": {"type": "number"}}}
    assert schema_arg_errors({"n": 1.5, "other": "x"}, open_schema) == []
    # Union types, anyOf and unknown type keywords.
    assert schema_arg_errors({"n": None}, {"properties": {"n": {"type": ["integer", "null"]}}}) == []
    any_of = {"properties": {"n": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}
    assert schema_arg_errors({"n": 3}, any_of) == []
    assert schema_arg_errors({"n": 3.5}, any_of) == ["argument 'n' has invalid value 3.5"]
    assert schema_arg_errors({"n": "x"}, {"properties": {"n": {"type": "custom"}}}) == []


# --- efficiency -------------------------------------------------------------


def test_efficiency_uses_expected_count_by_default():
    calls = [_call("get_part")] * 4
    s = ToolUseScorer(min_efficiency=0.5).score(_case(["get_part"]), _run(calls))
    assert s.details["optimal_calls"] == 1
    assert s.details["efficiency"] == pytest.approx(0.25)
    assert not s.passed and "efficiency" in s.details["failed_checks"]


def test_optimal_steps_overrides_expected_count():
    case = _case(["search_parts", "check_stock"], optimal_steps=3)
    calls = [_call("search_parts"), _call("list_warehouses"), _call("check_stock")]
    s = ToolUseScorer(min_efficiency=1.0).score(case, _run(calls))
    assert s.details["optimal_calls"] == 3
    assert s.details["efficiency"] == 1.0
    assert s.passed


def test_fewer_calls_than_optimal_is_capped():
    s = ToolUseScorer(min_recall=0.0).score(_case(["a", "b"], optimal_steps=2), _run([_call("a")]))
    assert s.details["efficiency"] == 1.0
    assert s.details["recall"] == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [-1, 1.5, "2", True])
def test_bad_optimal_steps_fails_case(bad):
    s = ToolUseScorer().score(_case(["a"], optimal_steps=bad), _run([_call("a")]))
    assert not s.passed and s.score == 0.0
    assert "optimal_steps" in s.details["error"]


def test_custom_metadata_and_trace_keys():
    scorer = ToolUseScorer(trace_key="calls", expected_key="want", optimal_key="best")
    case = EvalCase(id="c1", input="q", metadata={"want": ["a"], "best": 1})
    run = RunResult(case_id="c1", output="x", trace={"calls": [_call("a"), _call("a")]})
    s = scorer.score(case, run)
    assert s.details["efficiency"] == pytest.approx(0.5)
    assert s.details["precision"] == pytest.approx(0.5)


# --- thresholds and score ---------------------------------------------------


def test_all_failed_checks_are_reported():
    scorer = ToolUseScorer(min_precision=0.9, min_recall=0.9, max_invalid_arg_rate=0.0, min_efficiency=0.9)
    calls = [_call("a", error="validation error"), _call("x"), _call("y")]
    s = scorer.score(_case(["a", "b"]), _run(calls))
    assert s.details["failed_checks"] == ["precision", "recall", "invalid_arg_rate", "efficiency"]
    assert s.details["thresholds"] == {
        "min_precision": 0.9,
        "min_recall": 0.9,
        "max_invalid_arg_rate": 0.0,
        "min_efficiency": 0.9,
    }


def test_score_is_mean_of_components():
    calls = [_call("a"), _call("a", error="invalid arguments"), _call("z")]
    s = ToolUseScorer(min_recall=0.0, max_invalid_arg_rate=1.0).score(_case(["a", "b"]), _run(calls))
    d = s.details
    expected = (d["precision"] + d["recall"] + (1 - d["invalid_arg_rate"]) + d["efficiency"]) / 4
    assert s.score == pytest.approx(expected)
    assert (d["precision"], d["recall"], d["invalid_arg_rate"], d["efficiency"]) == pytest.approx(
        (1 / 3, 1 / 2, 1 / 3, 2 / 3)
    )
    assert 0.0 <= s.score <= 1.0


def test_thresholds_are_inclusive():
    calls = [_call("a"), _call("b")]
    s = ToolUseScorer(min_precision=0.5, min_recall=1.0, min_efficiency=0.5).score(_case(["a"]), _run(calls))
    assert (s.details["precision"], s.details["efficiency"]) == (0.5, 0.5)
    assert s.passed


# --- skips, run errors and malformed input ----------------------------------


def test_skips_without_expected_tools():
    s = ToolUseScorer().score(_case(), _run([_call("a")]))
    assert s.passed and s.score == 1.0
    assert "expected_tools" in s.details["skipped"]


def test_skips_without_trace_tool_calls():
    s = ToolUseScorer().score(_case(["a"]), _run())
    assert s.passed and "tool_calls" in s.details["skipped"]


def test_run_error_fails_but_reports_metrics():
    run = _run([_call("a"), _call("a")], error="max_steps (2) reached without a final answer")
    s = ToolUseScorer().score(_case(["a"]), run)
    assert not s.passed
    assert s.details["error"].startswith("max_steps")
    assert s.details["recall"] == 1.0 and s.details["precision"] == pytest.approx(0.5)
    assert s.details["failed_checks"] == []  # the metrics alone would pass


@pytest.mark.parametrize("missing", ["expected", "trace"])
def test_run_error_without_data_fails_instead_of_skipping(missing):
    case = _case() if missing == "expected" else _case(["a"])
    run = _run(None if missing == "trace" else [_call("a")], error="ConnectionError: boom")
    s = ToolUseScorer().score(case, run)
    assert not s.passed and s.score == 0.0
    assert s.details["error"] == "ConnectionError: boom"


@pytest.mark.parametrize(
    "calls", ["get_part", {"name": "get_part"}, [42], [{"args": {}}], [{"name": "a", "args": "not json"}]]
)
def test_malformed_tool_calls_fail(calls):
    s = ToolUseScorer().score(_case(["get_part"]), _run(calls))
    assert not s.passed and s.score == 0.0
    assert "invalid tool calls" in s.details["error"]


def test_malformed_expected_tools_fail():
    s = ToolUseScorer().score(_case("get_part"), _run([_call("get_part")]))
    assert not s.passed and "invalid tool calls" in s.details["error"]


def test_accepts_openai_style_and_json_string_arguments():
    calls = [
        {"function": {"name": "get_part", "arguments": '{"sku": "BRK-001"}'}},
        {"name": "check_stock", "arguments": ""},
        "list_warehouses",
    ]
    case = _case([{"name": "get_part", "args": {"sku": "BRK-001"}}, "check_stock", "list_warehouses"])
    s = ToolUseScorer(match_args=True).score(case, _run(calls))
    assert s.passed and s.score == 1.0


# --- end to end -------------------------------------------------------------


def test_mcp_suite_tool_use_scores_with_mock_adapter(tmp_path):
    _, summary, outcomes = run_suite(
        suite_path=ROOT / "configs" / "mcp_suite.yaml",
        project_root=ROOT,
        adapter_name="mock",
        store=RunStore(tmp_path / "runs.db"),
    )
    assert summary.pass_rate == 1.0
    tool_use = {o.case_id: next(s for s in o.scores if s.scorer == "tool_use") for o in outcomes}
    assert len(tool_use) == 8
    assert tool_use["mcp-part-price"].score == 1.0
    # One extra list_warehouses call: half precision and efficiency, still within the suite's thresholds.
    extra = tool_use["mcp-stock-one-warehouse"]
    assert extra.passed
    assert (extra.details["precision"], extra.details["efficiency"]) == (0.5, 0.5)
    assert tool_use["mcp-no-tools"].details["calls"] == 0
