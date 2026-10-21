from __future__ import annotations

import json
from pathlib import Path

import yaml

from ai_eval.models import EvalCase, SuiteConfig


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                cases.append(EvalCase.model_validate(data))
            except Exception as e:
                raise ValueError(f"{path}:{line_no}: invalid case: {e}") from e
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def load_suite(path: Path, project_root: Path | None = None) -> SuiteConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    suite = SuiteConfig.model_validate(raw)
    dataset_path = Path(suite.dataset)
    if not dataset_path.is_absolute() and project_root:
        dataset_path = (project_root / dataset_path).resolve()
    suite.dataset = str(dataset_path)
    return suite
