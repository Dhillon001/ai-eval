"""`compare --alpha` / `--min-effect`: only significant, large-enough regressions fail the gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_eval.cli import app
from ai_eval.models import (
    CaseOutcome,
    EvalCase,
    EvalRunSummary,
    RunExport,
    RunResult,
    ScoreResult,
    utc_now,
)
from ai_eval.report import compare_exports, regression_gate
from ai_eval.store import save_run_file

runner = CliRunner()
N = 40


def _outcome(cid: str, score: float) -> CaseOutcome:
    return CaseOutcome(
        case_id=cid,
        run=RunResult(case_id=cid, output="o"),
        scores=[ScoreResult(scorer="judge", passed=score >= 0.5, score=score)],
    )


def _export(run_id: str, outcomes: list[CaseOutcome]) -> RunExport:
    passed = sum(o.passed for o in outcomes)
    return RunExport(
        summary=EvalRunSummary(
            run_id=run_id, suite_name="s", created_at=utc_now(), adapter="mock", model=None,
            config_hash="h", git_sha=None, total=len(outcomes), passed=passed,
            pass_rate=passed / len(outcomes), mean_score=0.5,
        ),
        outcomes=outcomes,
        cases=[EvalCase(id=o.case_id, input="q") for o in outcomes],
    )


def _base() -> list[CaseOutcome]:
    return [_outcome(f"c{i}", 0.6 + 0.3 * (i % 5) / 4) for i in range(N)]


def _with(base: list[CaseOutcome], overrides: dict[int, float]) -> list[CaseOutcome]:
    return [_outcome(o.case_id, overrides.get(i, o.scores[0].score)) for i, o in enumerate(base)]


def _diff(overrides: dict[int, float]) -> dict:
    base = _base()
    return compare_exports(_export("A", base), _export("B", _with(base, overrides)))


ONE_FLIP = {0: 0.1}  # a single regression: real but not significant
MANY_FLIPS = {i: 0.1 for i in range(12)}  # 12 of 40 cases regress: p << 0.05, drop 0.30


# --- regression_gate ----------------------------------------------------------------------------


def test_strict_default_fails_on_any_regression():
    gate = regression_gate(_diff(ONE_FLIP))
    assert gate["failed"] and gate["reasons"] == ["1 regression(s)"]
    assert gate["alpha"] is None and gate["min_effect"] is None


def test_no_change_never_fails():
    gate = regression_gate(_diff({}), alpha=0.05, min_effect=0.0, fail_on_score_drop=True)
    assert not gate["failed"] and gate["reasons"] == [] and gate["ignored"] == []


def test_alpha_waives_a_single_noisy_flip():
    gate = regression_gate(_diff(ONE_FLIP), alpha=0.05)
    assert not gate["failed"]
    assert len(gate["ignored"]) == 1 and "alpha 0.05" in gate["ignored"][0]


def test_alpha_keeps_a_clear_regression():
    gate = regression_gate(_diff(MANY_FLIPS), alpha=0.05)
    assert gate["failed"]
    assert gate["reasons"][0].startswith("12 regression(s)") and "p=" in gate["reasons"][0]


def test_min_effect_filters_small_drops():
    diff = _diff(MANY_FLIPS)  # pass rate drops by 12/40 = 0.30
    assert regression_gate(diff, min_effect=0.3)["failed"]  # boundary counts as a fail
    small = regression_gate(diff, min_effect=0.31)
    assert not small["failed"] and "min effect 0.31" in small["ignored"][0]


def test_alpha_and_min_effect_both_required():
    diff = _diff(MANY_FLIPS)
    assert regression_gate(diff, alpha=0.05, min_effect=0.1)["failed"]
    assert not regression_gate(diff, alpha=0.05, min_effect=0.5)["failed"]
    assert not regression_gate(_diff(ONE_FLIP), alpha=0.05, min_effect=0.0)["failed"]


def test_regressions_outweighed_by_improvements_do_not_fail():
    base = _base()
    base[1] = _outcome("c1", 0.1)
    base[2] = _outcome("c2", 0.1)
    cand = _with(base, {0: 0.1, 1: 0.9, 2: 0.9})  # 1 regression, 2 improvements
    diff = compare_exports(_export("A", base), _export("B", cand))
    assert diff["regressions"] == ["c0"]
    gate = regression_gate(diff, min_effect=0.0)
    assert not gate["failed"] and "not a drop" in gate["ignored"][0]
    assert regression_gate(diff)["failed"]  # strict mode still fails on any regression


def test_score_drops_need_flag_and_respect_alpha():
    small = _diff({4: 0.5})  # one case still passing, score drop > 0.1
    assert small["score_drops"] and not small["regressions"]
    assert not regression_gate(small)["failed"]
    assert regression_gate(small, fail_on_score_drop=True)["failed"]
    assert not regression_gate(small, fail_on_score_drop=True, alpha=0.05)["failed"]

    broad = _diff({i: 0.55 for i in range(N)})  # every case drops: significant score delta
    gate = regression_gate(broad, fail_on_score_drop=True, alpha=0.05, min_effect=0.05)
    assert gate["failed"] and gate["reasons"][0].endswith(")") and "score drop(s)" in gate["reasons"][0]


@pytest.mark.parametrize("kw", [{"alpha": 0.0}, {"alpha": 1.5}, {"min_effect": -0.1}])
def test_invalid_thresholds_raise(kw):
    with pytest.raises(ValueError):
        regression_gate(_diff({}), **kw)


# --- CLI ----------------------------------------------------------------------------------------


@pytest.fixture
def files(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _base()
    save_run_file(_export("A", base), tmp_path / "a.json")
    save_run_file(_export("one", _with(base, ONE_FLIP)), tmp_path / "one.json")
    save_run_file(_export("many", _with(base, MANY_FLIPS)), tmp_path / "many.json")
    return tmp_path


def _compare(cand: str, *args: str):
    return runner.invoke(app, ["compare", "--baseline-file", "a.json", "--candidate-file", cand, *args])


def test_cli_alpha_lets_noise_pass(files):
    assert _compare("one.json").exit_code == 1
    res = _compare("one.json", "--alpha", "0.05")
    assert res.exit_code == 0, res.output
    gate = json.loads(res.output)["gate"]
    assert gate["failed"] is False and gate["alpha"] == 0.05 and gate["ignored"]


def test_cli_alpha_still_fails_on_real_regression(files):
    res = _compare("many.json", "--alpha", "0.05", "--min-effect", "0.1")
    assert res.exit_code == 1, res.output
    data = json.loads(res.output)
    assert data["gate"]["failed"] and data["gate"]["min_effect"] == 0.1
    assert len(data["regressions"]) == 12


def test_cli_min_effect_alone(files):
    assert _compare("many.json", "--min-effect", "0.5").exit_code == 0
    assert _compare("many.json", "--min-effect", "0.2").exit_code == 1


@pytest.mark.parametrize("args", [("--alpha", "0"), ("--alpha", "2"), ("--min-effect", "-1")])
def test_cli_rejects_bad_thresholds(files, args):
    assert _compare("one.json", *args).exit_code == 2
