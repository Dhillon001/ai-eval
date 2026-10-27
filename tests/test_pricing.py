import json

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from ai_eval.cli import app
from ai_eval.models import CaseOutcome, RunResult, ScoreResult
from ai_eval.pricing import DEFAULT_PRICES, ModelPrice, PriceTable, run_cost
from fakes import FakeOpenAI

# Round numbers keep the expected costs exact.
SIMPLE = ModelPrice(input_per_mtok=1.0, output_per_mtok=2.0)
TIERED = ModelPrice(
    input_per_mtok=1.0,
    output_per_mtok=2.0,
    long_context_threshold=1000,
    long_input_per_mtok=10.0,
    long_output_per_mtok=20.0,
)


def test_cost_per_call():
    assert SIMPLE.cost(1_000_000, 500_000) == pytest.approx(2.0)
    assert SIMPLE.cost(300, None) == pytest.approx(0.0003)
    assert SIMPLE.cost(None, None) == 0.0


def test_long_context_tier_bills_whole_call_above_threshold():
    assert TIERED.cost(1000, 100) == pytest.approx((1000 * 1 + 100 * 2) / 1e6)  # at threshold: base
    assert TIERED.cost(1001, 100) == pytest.approx((1001 * 10 + 100 * 20) / 1e6)


def test_builtin_haiku_long_context_tier():
    haiku = DEFAULT_PRICES["claude-haiku-5-5"]
    assert haiku.cost(100_000, 0) == pytest.approx(0.01)
    assert haiku.cost(200_000, 0) == pytest.approx(0.10)


@pytest.mark.parametrize(
    "spec",
    [
        {"input_per_mtok": -1, "output_per_mtok": 1},
        {"input_per_mtok": 1, "output_per_mtok": 1, "long_context_threshold": 10},
        {"input_per_mtok": 1, "output_per_mtok": 1, "long_input_per_mtok": 2, "long_output_per_mtok": 2},
    ],
)
def test_invalid_prices_are_rejected(spec):
    with pytest.raises(ValidationError):
        ModelPrice.model_validate(spec)


def test_lookup_exact_snapshot_and_longest_prefix():
    table = PriceTable()
    assert table.lookup("gpt-4o") is DEFAULT_PRICES["gpt-4o"]
    assert table.lookup("gpt-4o-2024-08-06") is DEFAULT_PRICES["gpt-4o"]
    assert table.lookup("gpt-4o-mini-2024-07-18") is DEFAULT_PRICES["gpt-4o-mini"]
    assert table.lookup("gpt-4oz") is None  # prefix must be followed by "-"
    assert table.lookup("unknown") is None and table.lookup(None) is None
    assert table.cost("unknown", 10, 10) is None


def test_suite_overrides_merge_without_mutating_defaults():
    original = DEFAULT_PRICES["gpt-4o"]
    table = PriceTable.from_config(
        {"gpt-4o": {"input_per_mtok": 1, "output_per_mtok": 2}, "my-model": SIMPLE}
    )
    assert table.lookup("gpt-4o").input_per_mtok == 1
    assert table.lookup("my-model") is SIMPLE
    assert table.lookup("gpt-4o-mini") is DEFAULT_PRICES["gpt-4o-mini"]
    assert DEFAULT_PRICES["gpt-4o"] is original and PriceTable().lookup("gpt-4o") is original


def _judge(prompt, completion, model="judge-m", cached=False):
    return ScoreResult(
        scorer="llm_judge",
        passed=True,
        score=1.0,
        details={"judge": f"openai:{model}", "cached": cached, "prompt_tokens": prompt, "completion_tokens": completion},
    )


def _outcome(prompt=None, completion=None, model=None, scores=()):
    trace = {"model": model} if model else {}
    run = RunResult(case_id="1", output="a", trace=trace, prompt_tokens=prompt, completion_tokens=completion)
    return CaseOutcome(case_id="1", run=run, scores=list(scores))


def test_run_cost_splits_system_and_judge_and_skips_cached_verdicts():
    table = PriceTable({"sys-m": SIMPLE, "judge-m": ModelPrice(input_per_mtok=4.0, output_per_mtok=8.0)})
    outcomes = [
        _outcome(1000, 500, model="sys-m", scores=[_judge(100, 50)]),  # sys 0.002, judge 0.0008
        _outcome(1000, 500, model="sys-m", scores=[_judge(100, 50, cached=True)]),  # judge free
    ]
    cost = run_cost(outcomes, table)
    assert cost["cost_usd_system"] == pytest.approx(0.004)
    assert cost["cost_usd_judge"] == pytest.approx(0.0008)
    assert cost["cost_usd"] == pytest.approx(0.0048)
    assert cost["unpriced_models"] == []


def test_run_cost_uses_default_model_and_lists_unpriced():
    table = PriceTable({"sys-m": SIMPLE})
    outcomes = [_outcome(1000, 0), _outcome(10, 10, model="mystery", scores=[_judge(1, 1, model="j?")])]
    cost = run_cost(outcomes, table, default_model="sys-m")
    assert cost["cost_usd"] == pytest.approx(0.001)
    assert cost["cost_usd_judge"] is None
    assert cost["unpriced_models"] == ["j?", "mystery"]


def test_run_cost_is_none_without_token_counts():
    cost = run_cost([_outcome(scores=[_judge(None, None)])], PriceTable())
    assert cost == {"cost_usd": None, "cost_usd_system": None, "cost_usd_judge": None, "unpriced_models": []}


def test_cli_run_reports_cost_with_suite_pricing(tmp_path, monkeypatch):
    client = FakeOpenAI('{"pass": true, "score": 1.0}')  # reports 10 prompt + 5 completion tokens
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ai_eval.scorers.llm_judge.make_client", lambda provider: client)
    (tmp_path / "data.jsonl").write_text(
        "".join(json.dumps({"id": str(i), "input": "q", "expected": "a"}) + "\n" for i in range(2))
    )
    suite = {
        "name": "priced",
        "dataset": "data.jsonl",
        "adapter": {"type": "mock"},
        "scorers": [{"type": "llm_judge", "provider": "openai", "model": "gpt-4o-mini"}],
        "pricing": {"gpt-4o-mini": {"input_per_mtok": 1000, "output_per_mtok": 2000}},
    }
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump(suite))

    result = CliRunner().invoke(app, ["run", "--suite", "suite.yaml", "--no-cache"])
    assert result.exit_code == 0, result.output
    # per judge call: (10 * 1000 + 5 * 2000) / 1e6 = $0.02; two cases
    assert "Cost: $0.0400 (system $0.0000 + judge $0.0400)" in result.output
