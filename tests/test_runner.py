from pathlib import Path

from ai_eval.runner import run_suite
from ai_eval.store import RunStore


def test_mock_run(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    run_id, summary, outcomes = run_suite(
        suite_path=root / "configs" / "example_suite.yaml",
        project_root=root,
        adapter_name="mock",
        store=RunStore(tmp_path / "test.db"),
    )
    assert summary.total == 3
    assert summary.pass_rate >= 2 / 3
    assert len(outcomes) == 3


def test_mock_run_with_repeats(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    store = RunStore(tmp_path / "test.db")
    run_id, summary, outcomes = run_suite(
        suite_path=root / "configs" / "example_suite.yaml",
        project_root=root,
        adapter_name="mock",
        store=store,
        repeats=3,
    )
    assert summary.total == 3
    assert summary.metadata["repeats"] == 3
    assert summary.metadata["samples"] == 9
    for o in outcomes:
        assert o.repeats == 3 and len(o.samples) == 3
        assert o.consistency == 1.0  # the mock adapter is deterministic
    stored = {o.case_id: o for o in store.get_outcomes(run_id)}
    assert all(len(o.samples) == 3 for o in stored.values())
