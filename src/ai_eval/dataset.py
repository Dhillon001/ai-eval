from __future__ import annotations

import json
from pathlib import Path

import yaml

from ai_eval.models import EvalCase, HumanLabel, SuiteConfig


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


def load_labels(path: Path) -> dict[str, HumanLabel]:
    """Load human labels from a JSONL file, one object per line keyed by case id.

    Each line is ``{"id": "<case id>", "passed": true, "score": 0.8, "notes": "..."}``
    (``case_id`` is accepted for ``id`` and ``pass`` for ``passed``; ``score``/``notes`` are
    optional). Blank lines and ``#`` comments are skipped. Duplicate ids are an error.
    """
    labels: dict[str, HumanLabel] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise TypeError("expected a JSON object")
                if "id" in data and "case_id" in data:
                    raise ValueError("use either 'id' or 'case_id', not both")
                case_id = data.pop("id", data.pop("case_id", None))
                if not isinstance(case_id, str) or not case_id:
                    raise ValueError("missing case 'id'")
                label = HumanLabel.model_validate(data)
            except Exception as e:
                raise ValueError(f"{path}:{line_no}: invalid label: {e}") from e
            if case_id in labels:
                raise ValueError(f"{path}:{line_no}: duplicate label for case {case_id!r}")
            labels[case_id] = label
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def apply_labels(cases: list[EvalCase], labels: dict[str, HumanLabel]) -> list[EvalCase]:
    """Return copies of ``cases`` with ``human_label`` set from ``labels``.

    Labels from the file take precedence over inline ``human_label`` values. A label whose id
    matches no case is an error (usually a typo or a stale labels file). Cases without a label
    keep their inline label, or ``None``.
    """
    known = {c.id for c in cases}
    unknown = sorted(set(labels) - known)
    if unknown:
        shown = ", ".join(repr(u) for u in unknown[:5])
        more = f" (+{len(unknown) - 5} more)" if len(unknown) > 5 else ""
        raise ValueError(f"Labels reference unknown case ids: {shown}{more}")
    return [
        c.model_copy(update={"human_label": labels[c.id]}) if c.id in labels else c for c in cases
    ]


def labeled_cases(cases: list[EvalCase]) -> list[EvalCase]:
    """The subset of ``cases`` that carry a human label."""
    return [c for c in cases if c.human_label is not None]
