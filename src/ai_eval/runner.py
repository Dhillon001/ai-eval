from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path

from ai_eval.adapters import build_adapter
from ai_eval.dataset import load_cases, load_suite
from ai_eval.models import CaseOutcome, EvalRunSummary, utc_now
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


def run_suite(
    suite_path: Path,
    project_root: Path,
    adapter_name: str | None = None,
    adapter_options: dict | None = None,
    store: RunStore | None = None,
) -> tuple[str, EvalRunSummary, list[CaseOutcome]]:
    suite = load_suite(suite_path, project_root=project_root)
    cases = load_cases(Path(suite.dataset))

    adapter_cfg = dict(suite.adapter)
    name = adapter_name or adapter_cfg.pop("type", "mock")
    if adapter_options:
        adapter_cfg.update(adapter_options)
    adapter = build_adapter(name, adapter_cfg)

    scorers = [build_scorer(s) for s in suite.scorers]
    outcomes: list[CaseOutcome] = []

    for case in cases:
        run = adapter.run(case)
        scores = [scorer.score(case, run) for scorer in scorers]
        outcomes.append(CaseOutcome(case_id=case.id, run=run, scores=scores))

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
        metadata={"suite_path": str(suite_path)},
    )

    if store is None:
        store = RunStore.default(project_root)
    store.save_run(summary, outcomes, cases)
    return run_id, summary, outcomes
