from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_eval.cli import app
from ai_eval.models import RUN_EXPORT_FORMAT_VERSION, RunExport
from ai_eval.report import compare_exports
from ai_eval.runner import run_suite
from ai_eval.store import RunStore, load_run_file, save_run_file

ROOT = Path(__file__).resolve().parents[1]
SUITE = "configs/example_suite.yaml"

runner = CliRunner()


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A throwaway project root with the example suite, so the CLI's .ai-eval/ lands in tmp."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "datasets", tmp_path / "datasets")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _mock_export(project: Path) -> RunExport:
    store = RunStore.default(project)
    run_id, _, _ = run_suite(
        suite_path=project / SUITE, project_root=project, adapter_name="mock", store=store
    )
    return store.export_run(run_id)


def _degrade(export: RunExport, case_id: str, *, passed: bool, score: float) -> RunExport:
    data = export.model_copy(deep=True)
    for o in data.outcomes:
        if o.case_id == case_id:
            assert o.scores, "test expects the case to have scores"
            for s in o.scores:
                s.passed, s.score = passed, score
    return data


def _passing_scored_case(export: RunExport) -> str:
    return next(o.case_id for o in export.outcomes if o.passed and o.scores)


# --- store-level round trip -------------------------------------------------


def test_export_import_round_trip(project):
    export = _mock_export(project)
    path = save_run_file(export, project / "out" / "nested" / "base.json")
    assert path.is_file()

    loaded = load_run_file(path)
    assert loaded == export
    assert loaded.format_version == RUN_EXPORT_FORMAT_VERSION
    assert loaded.summary.total == len(loaded.outcomes) == len(loaded.cases) == 3
    assert set(loaded.cases_by_id()) == {o.case_id for o in loaded.outcomes}


def test_export_unknown_run_raises(tmp_path):
    with pytest.raises(ValueError, match="Run not found"):
        RunStore(tmp_path / "runs.db").export_run("nope")


def test_load_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_run_file(tmp_path / "missing.json")


def test_load_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid run file"):
        load_run_file(p)


def test_load_rejects_newer_format(project):
    export = _mock_export(project)
    data = json.loads(export.model_dump_json())
    data["format_version"] = RUN_EXPORT_FORMAT_VERSION + 1
    p = project / "future.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        load_run_file(p)


# --- compare against an imported baseline -----------------------------------


def test_compare_identical_imported_baseline(project):
    export = _mock_export(project)
    baseline = load_run_file(save_run_file(export, project / "base.json"))
    result = compare_exports(baseline, export)
    assert result["regressions"] == []
    assert result["score_drops"] == []
    assert result["improvements"] == []
    assert result["pass_rate_delta"] == 0


def test_compare_imported_baseline_detects_regression_and_improvement(project):
    export = _mock_export(project)
    cid = _passing_scored_case(export)
    baseline = load_run_file(save_run_file(export, project / "base.json"))

    regressed = _degrade(export, cid, passed=False, score=0.0)
    assert compare_exports(baseline, regressed)["regressions"] == [cid]
    # Swapping sides turns it into an improvement.
    assert compare_exports(regressed, baseline)["improvements"] == [cid]


# --- CLI exit codes ---------------------------------------------------------


def _cli_run_with_baseline(path: str) -> RunExport:
    res = runner.invoke(app, ["run", "--suite", SUITE, "--adapter", "mock", "--save-baseline", path])
    assert res.exit_code == 0, res.output
    return load_run_file(Path(path))


def _compare(*args: str):
    return runner.invoke(app, ["compare", *args])


def test_cli_save_baseline_writes_file(project):
    export = _cli_run_with_baseline("baselines/main.json")
    assert (project / "baselines" / "main.json").is_file()
    assert export.summary.total == 3
    # The exported run id is the one stored locally.
    assert RunStore.default(project).get_run(export.summary.run_id) is not None


def test_cli_compare_files_no_regression_exits_0(project):
    _cli_run_with_baseline("base.json")
    res = _compare("--baseline-file", "base.json", "--candidate-file", "base.json")
    assert res.exit_code == 0, res.output
    assert '"regressions": []' in res.output


def test_cli_compare_file_vs_store_run(project):
    export = _cli_run_with_baseline("base.json")
    res = _compare("--baseline-file", "base.json", "--candidate", export.summary.run_id)
    assert res.exit_code == 0, res.output


def test_cli_compare_regression_exits_1(project):
    export = _cli_run_with_baseline("base.json")
    cid = _passing_scored_case(export)
    save_run_file(_degrade(export, cid, passed=False, score=0.0), project / "cand.json")
    res = _compare("--baseline-file", "base.json", "--candidate-file", "cand.json")
    assert res.exit_code == 1
    assert cid in res.output


def test_cli_compare_score_drop_needs_flag(project):
    export = _cli_run_with_baseline("base.json")
    cid = _passing_scored_case(export)
    # Still passing, but the score drops well past the 0.1 threshold.
    save_run_file(_degrade(export, cid, passed=True, score=0.5), project / "cand.json")
    args = ("--baseline-file", "base.json", "--candidate-file", "cand.json")

    assert _compare(*args).exit_code == 0
    res = _compare(*args, "--fail-on-score-drop")
    assert res.exit_code == 1
    assert cid in res.output


@pytest.mark.parametrize(
    "args",
    [
        ("--baseline-file", "missing.json", "--candidate-file", "base.json"),
        ("--baseline-file", "base.json", "--candidate-file", "missing.json"),
        ("--baseline-file", "bad.json", "--candidate-file", "base.json"),
        ("--baseline", "no-such-run", "--candidate-file", "base.json"),
        ("--candidate-file", "base.json"),  # no baseline at all
        ("--baseline", "x", "--baseline-file", "base.json", "--candidate-file", "base.json"),  # both
    ],
)
def test_cli_compare_bad_inputs_exit_2(project, args):
    _cli_run_with_baseline("base.json")
    (project / "bad.json").write_text("{not json", encoding="utf-8")
    res = _compare(*args)
    assert res.exit_code == 2, res.output
