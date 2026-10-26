from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from ai_eval.adapters import build_adapter
from ai_eval.cache import JudgeCache
from ai_eval.dataset import load_cases, load_suite
from ai_eval.adapters.base import SystemAdapter
from ai_eval.models import CaseOutcome, EvalCase, EvalRunSummary, RunResult, ScoreResult, utc_now
from ai_eval.scorers.base import Scorer
from ai_eval.scorers import build_scorer
from ai_eval.store import RunStore


def config_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def git_sha(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def evaluate_case(case: EvalCase, adapter: SystemAdapter, scorers: list[Scorer]) -> CaseOutcome:
    """Run one case end to end. Never raises: adapter/scorer crashes become recorded failures."""
    try:
        run = adapter.run(case)
    except Exception as e:
        run = RunResult(case_id=case.id, output="", error=f"adapter crashed: {type(e).__name__}: {e}")
    scores: list[ScoreResult] = []
    for scorer in scorers:
        try:
            scores.append(scorer.score(case, run))
        except Exception as e:
            scores.append(
                ScoreResult(
                    scorer=getattr(scorer, "name", type(scorer).__name__),
                    passed=False,
                    score=0.0,
                    details={"error": f"scorer crashed: {type(e).__name__}: {e}"},
                )
            )
    return CaseOutcome(case_id=case.id, run=run, scores=scores)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def run_stats(outcomes: list[CaseOutcome]) -> dict:
    latencies = [o.run.latency_ms for o in outcomes]
    prompt = [o.run.prompt_tokens for o in outcomes if o.run.prompt_tokens is not None]
    completion = [o.run.completion_tokens for o in outcomes if o.run.completion_tokens is not None]
    return {
        "errors": sum(1 for o in outcomes if o.run.error),
        "latency_ms_p50": round(_percentile(latencies, 50), 2),
        "latency_ms_p95": round(_percentile(latencies, 95), 2),
        "prompt_tokens": sum(prompt) if prompt else None,
        "completion_tokens": sum(completion) if completion else None,
        "judge_cache_hits": sum(1 for o in outcomes for s in o.scores if s.details.get("cached") is True),
    }


def run_suite(
    suite_path: Path,
    project_root: Path,
    adapter_name: str | None = None,
    adapter_options: dict | None = None,
    store: RunStore | None = None,
    concurrency: int = 1,
    on_case_done: Callable[[CaseOutcome], None] | None = None,
    use_cache: bool = True,
    cache: JudgeCache | None = None,
) -> tuple[str, EvalRunSummary, list[CaseOutcome]]:
    suite = load_suite(suite_path, project_root=project_root)
    cases = load_cases(Path(suite.dataset))

    adapter_cfg = dict(suite.adapter)
    name = adapter_name or adapter_cfg.pop("type", "mock")
    if adapter_options:
        adapter_cfg.update(adapter_options)
    adapter = build_adapter(name, adapter_cfg)

    if use_cache and cache is None:
        cache = JudgeCache.default(project_root)
    scorers = [build_scorer(s, cache=cache if use_cache else None) for s in suite.scorers]

    def _one(case: EvalCase) -> CaseOutcome:
        outcome = evaluate_case(case, adapter, scorers)
        if on_case_done:
            on_case_done(outcome)
        return outcome

    if concurrency <= 1:
        outcomes = [_one(c) for c in cases]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outcomes = list(pool.map(_one, cases))  # map preserves dataset order

    passed = sum(1 for o in outcomes if o.passed)
    total = len(outcomes)
    all_scores = [s.score for o in outcomes for s in o.scores]
    mean_score = sum(all_scores) / len(all_scores) if all_scores else (passed / total if total else 0.0)

    cfg_payload = {
        "suite": suite.model_dump(),
        "adapter": name,
        "adapter_options": adapter_cfg,
    }
    run_id = uuid.uuid4().hex[:12]
    summary = EvalRunSummary(
        run_id=run_id,
        suite_name=suite.name,
        created_at=utc_now(),
        adapter=name,
        model=adapter_cfg.get("model"),
        config_hash=config_hash(cfg_payload),
        git_sha=git_sha(project_root),
        total=total,
        passed=passed,
        pass_rate=passed / total if total else 0.0,
        mean_score=mean_score,
        metadata={"suite_path": str(suite_path), "concurrency": concurrency, **run_stats(outcomes)},
    )

    if store is None:
        store = RunStore.default(project_root)
    store.save_run(summary, outcomes, cases)
    return run_id, summary, outcomes
