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
