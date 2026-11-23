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
from ai_eval.models import CaseOutcome, CaseSample, EvalCase, EvalRunSummary, RunResult, ScoreResult, utc_now
from ai_eval.pricing import PriceTable, run_cost
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


def merge_samples(samples: list[CaseOutcome]) -> CaseOutcome:
    """Fold N runs of one case into a single outcome (first sample on top, all N in ``samples``)."""
    if not samples:
        raise ValueError("merge_samples needs at least one sample")
    first = samples[0]
    if len(samples) == 1:
        return first
    return CaseOutcome(
        case_id=first.case_id,
        run=first.run,
        scores=first.scores,
        samples=[CaseSample(run=o.run, scores=o.scores) for o in samples],
    )


def run_stats(outcomes: list[CaseOutcome]) -> dict:
    """Latency, token, error and cache stats over every sample of every case."""
    samples = [smp for o in outcomes for smp in o.sample_list()]
    latencies = [smp.run.latency_ms for smp in samples]
    prompt = [smp.run.prompt_tokens for smp in samples if smp.run.prompt_tokens is not None]
    completion = [smp.run.completion_tokens for smp in samples if smp.run.completion_tokens is not None]
    stats = {
        "errors": sum(1 for smp in samples if smp.run.error),
        "latency_ms_p50": round(_percentile(latencies, 50), 2),
        "latency_ms_p95": round(_percentile(latencies, 95), 2),
        "prompt_tokens": sum(prompt) if prompt else None,
        "completion_tokens": sum(completion) if completion else None,
        "judge_cache_hits": sum(1 for smp in samples for s in smp.scores if s.details.get("cached") is True),
    }
    if any(o.samples for o in outcomes):
        n = len(outcomes)
        stats["samples"] = len(samples)
        stats["sample_pass_rate"] = round(sum(o.passed_samples for o in outcomes) / len(samples), 6) if samples else 0.0
        stats["mean_consistency"] = round(sum(o.consistency for o in outcomes) / n, 6) if n else 0.0
        # Flaky cases: repeats disagree, so their verdict could flip on a re-run.
        flaky = [o for o in outcomes if o.flaky]
        stats["flaky_count"] = len(flaky)
        stats["flaky_cases"] = [
            {"case_id": o.case_id, "passed_samples": o.passed_samples, "repeats": o.repeats, "passed": o.passed}
            for o in flaky
        ]
    return stats


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
    repeats: int = 1,
) -> tuple[str, EvalRunSummary, list[CaseOutcome]]:
    """Run a suite and store it. With ``repeats`` > 1 every case is run that many times; the
    samples are stored on each outcome and a case passes on a strict majority of its samples.
    ``on_case_done`` is called once per sample."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    suite = load_suite(suite_path, project_root=project_root)
    cases = load_cases(Path(suite.dataset))

    adapter_cfg = dict(suite.adapter)
    suite_adapter = adapter_cfg.pop("type", "mock")
    name = adapter_name or suite_adapter
    if name != suite_adapter:
        # Options in the suite belong to its own adapter (e.g. an http url, a mock template);
        # don't leak them into a different adapter chosen on the command line.
        adapter_cfg = {}
    if adapter_options:
        adapter_cfg.update(adapter_options)
    build_cfg = dict(adapter_cfg)
    if name == "mcp":
        # Relative server paths (cwd) resolve against the project root, like the dataset path.
        # Kept out of adapter_cfg so the stored config hash stays machine-independent.
        build_cfg.setdefault("root", str(project_root))
    adapter = build_adapter(name, build_cfg)
    prices = PriceTable.from_config(suite.pricing)

    if use_cache and cache is None:
        cache = JudgeCache.default(project_root)
    scorers = [build_scorer(s, cache=cache if use_cache else None) for s in suite.scorers]

    def _one(case: EvalCase) -> CaseOutcome:
        outcome = evaluate_case(case, adapter, scorers)
        if on_case_done:
            on_case_done(outcome)
        return outcome

    # Every (case, sample) pair is an independent job, so repeats parallelise like extra cases.
    jobs = [c for c in cases for _ in range(repeats)]
    try:
        if concurrency <= 1:
            flat = [_one(c) for c in jobs]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                flat = list(pool.map(_one, jobs))  # map preserves dataset order
    finally:
        adapter.close()
    outcomes = [merge_samples(flat[i : i + repeats]) for i in range(0, len(flat), repeats)]

    passed = sum(1 for o in outcomes if o.passed)
    total = len(outcomes)
    all_scores = [s.score for o in outcomes for smp in o.sample_list() for s in smp.scores]
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
        metadata={
            "suite_path": str(suite_path),
            "concurrency": concurrency,
            "repeats": repeats,
            **run_stats(outcomes),
            **run_cost(outcomes, prices, default_model=adapter_cfg.get("model")),
        },
    )

    if store is None:
        store = RunStore.default(project_root)
    store.save_run(summary, outcomes, cases)
    return run_id, summary, outcomes
