import threading
import time
from pathlib import Path

from ai_eval.adapters.base import SystemAdapter
from ai_eval.models import CaseOutcome, EvalCase, EvalRunSummary, RunResult, ScoreResult, utc_now
from ai_eval.report import compare_runs
from ai_eval.runner import evaluate_case, run_stats, run_suite
from ai_eval.scorers.checklist import ChecklistScorer
from ai_eval.store import RunStore


class Crashy(SystemAdapter):
    name = "crashy"

    def run(self, case):
        raise ValueError("kaboom")


def test_adapter_crash_is_isolated():
    o = evaluate_case(EvalCase(id="1", input="x", checks=["a"]), Crashy(), [ChecklistScorer()])
    assert not o.passed and "kaboom" in o.run.error


def test_concurrency_preserves_order_and_runs_parallel(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    active, peak, lock = [0], [0], threading.Lock()

    from ai_eval.adapters import mock

    orig = mock.MockAdapter.run

    def slow_run(self, case):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return orig(self, case)

    monkeypatch.setattr(mock.MockAdapter, "run", slow_run)
    _, summary, outcomes = run_suite(
        root / "configs" / "example_suite.yaml", root, store=RunStore(tmp_path / "t.db"), concurrency=3
    )
    assert [o.case_id for o in outcomes] == ["refund-1", "greeting-1", "edge-empty-checks"]
    assert peak[0] > 1
    assert summary.metadata["errors"] == 0 and "latency_ms_p95" in summary.metadata


def test_run_stats_tokens():
    outs = [CaseOutcome(case_id=str(i), run=RunResult(case_id=str(i), output="", latency_ms=i, prompt_tokens=2, completion_tokens=1)) for i in range(1, 5)]
    st = run_stats(outs)
    assert st["prompt_tokens"] == 8 and st["completion_tokens"] == 4 and st["latency_ms_p95"] == 4


def _summary(rid, rate):
    return EvalRunSummary(run_id=rid, suite_name="s", created_at=utc_now(), adapter="mock", model=None,
                          config_hash="h", git_sha=None, total=1, passed=1, pass_rate=rate, mean_score=rate)


def _out(cid, score, passed=True):
    return CaseOutcome(case_id=cid, run=RunResult(case_id=cid, output=""),
                       scores=[ScoreResult(scorer="checklist", passed=passed, score=score)])


def test_compare_flags_flips_drops_missing_and_tags():
    base = [_out("a", 1.0), _out("b", 1.0), _out("c", 0.0, passed=False), _out("gone", 1.0)]
    cand = [_out("a", 0.0, passed=False), _out("b", 0.5), _out("c", 1.0), _out("new", 1.0)]
    cases = {cid: EvalCase(id=cid, input="", metadata={"tags": ["billing"]}) for cid in "abc"}
    d = compare_runs(_summary("x", 0.75), _summary("y", 0.75), base, cand, cases, cases)
    assert d["regressions"] == ["a"]
    assert d["score_drops"] == [{"case_id": "b", "delta": -0.5}]
    assert d["improvements"] == ["c"]
    assert d["missing_in_candidate"] == ["gone"] and d["new_in_candidate"] == ["new"]
    assert "billing" in d["tag_pass_rate_deltas"]
