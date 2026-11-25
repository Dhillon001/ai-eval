"""Judge calibration: how well does the LLM judge agree with human labels?

``ai-eval calibrate`` runs a suite's system on every human-labeled case, grades the output with
the suite's LLM judge, and compares the judge's verdicts with the human ones:

- **accuracy** — fraction of cases where judge and human agree on pass/fail;
- **Cohen's kappa** — agreement corrected for chance, so a judge that passes everything on a
  mostly-passing set doesn't look good by accident (1 = perfect, 0 = chance, < 0 = worse);
- **confusion matrix** — with "pass" as the positive class: ``tp`` both pass, ``fp`` judge
  passes a human fail (the dangerous one), ``fn`` judge fails a human pass, ``tn`` both fail;
- **score correlation** — Pearson and Spearman between judge scores and human ``score`` grades,
  over the cases where the human gave one.

Cases without a human label are left out (and counted); cases where the judge errored or skipped
are reported but excluded from the metrics. Pure Python; no numpy needed.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel, Field

from ai_eval.adapters import build_adapter
from ai_eval.cache import JudgeCache
from ai_eval.dataset import apply_labels, labeled_cases, load_cases, load_labels, load_suite
from ai_eval.models import EvalCase, RunResult
from ai_eval.scorers import build_scorer
from ai_eval.scorers.base import Scorer

JUDGE_TYPES = ("llm_judge", "judge", "faithfulness")


class ConfusionMatrix(BaseModel):
    """Judge vs. human pass/fail counts, with "pass" as the positive class."""

    tp: int = 0  # judge pass, human pass
    fp: int = 0  # judge pass, human fail
    fn: int = 0  # judge fail, human pass
    tn: int = 0  # judge fail, human fail

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def add(self, judge_passed: bool, human_passed: bool) -> None:
        if judge_passed and human_passed:
            self.tp += 1
        elif judge_passed:
            self.fp += 1
        elif human_passed:
            self.fn += 1
        else:
            self.tn += 1


class CalibrationItem(BaseModel):
    """One labeled case: the human verdict next to the judge's."""

    case_id: str
    human_passed: bool
    human_score: float | None = None
    judge_passed: bool | None = None
    judge_score: float | None = None
    output: str = ""
    reason: str | None = None
    error: str | None = None  # adapter/judge error or skip; such items are excluded from metrics

    @property
    def scored(self) -> bool:
        return self.error is None and self.judge_passed is not None

    @property
    def agrees(self) -> bool | None:
        return None if not self.scored else self.judge_passed == self.human_passed


class CalibrationResult(BaseModel):
    suite_name: str
    judge: str
    labeled: int  # cases with a human label
    unlabeled: int  # cases skipped for lack of a label
    n: int  # labeled cases the judge actually graded (the metrics' denominator)
    accuracy: float | None = None
    kappa: float | None = None
    confusion: ConfusionMatrix = Field(default_factory=ConfusionMatrix)
    score_pairs: int = 0
    pearson: float | None = None
    spearman: float | None = None
    items: list[CalibrationItem] = Field(default_factory=list)

    @property
    def errors(self) -> list[CalibrationItem]:
        return [i for i in self.items if not i.scored]

    @property
    def disagreements(self) -> list[CalibrationItem]:
        return [i for i in self.items if i.agrees is False]


# --- metrics ---------------------------------------------------------------------------------


def confusion_matrix(pairs: Sequence[tuple[bool, bool]]) -> ConfusionMatrix:
    """Build a confusion matrix from ``(judge_passed, human_passed)`` pairs."""
    cm = ConfusionMatrix()
    for judge_passed, human_passed in pairs:
        cm.add(bool(judge_passed), bool(human_passed))
    return cm


def accuracy(cm: ConfusionMatrix) -> float | None:
    return (cm.tp + cm.tn) / cm.total if cm.total else None


def cohens_kappa(cm: ConfusionMatrix) -> float | None:
    """Cohen's kappa for two raters on a binary label.

    ``None`` when undefined: no pairs, or both raters used a single class (expected agreement
    is 1, so agreement beyond chance can't be measured — e.g. every case passes for both).
    """
    n = cm.total
    if not n:
        return None
    p_o = (cm.tp + cm.tn) / n
    judge_pass = (cm.tp + cm.fp) / n
    human_pass = (cm.tp + cm.fn) / n
    p_e = judge_pass * human_pass + (1 - judge_pass) * (1 - human_pass)
    if math.isclose(p_e, 1.0):
        return None
    return (p_o - p_e) / (1 - p_e)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation; ``None`` with fewer than 2 pairs or a constant series."""
    if len(xs) != len(ys):
        raise ValueError("pearson needs equal-length sequences")
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))


def _ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks with ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation (Pearson over tie-averaged ranks)."""
    if len(xs) != len(ys):
        raise ValueError("spearman needs equal-length sequences")
    return pearson(_ranks(xs), _ranks(ys))


def summarize(items: list[CalibrationItem], *, suite_name: str, judge: str, unlabeled: int = 0) -> CalibrationResult:
    """Compute agreement metrics over calibration items."""
    scored = [i for i in items if i.scored]
    cm = confusion_matrix([(bool(i.judge_passed), i.human_passed) for i in scored])
    with_scores = [i for i in scored if i.human_score is not None and i.judge_score is not None]
    xs = [float(i.judge_score) for i in with_scores]  # type: ignore[arg-type]
    ys = [float(i.human_score) for i in with_scores]  # type: ignore[arg-type]
    return CalibrationResult(
        suite_name=suite_name,
        judge=judge,
        labeled=len(items),
        unlabeled=unlabeled,
        n=len(scored),
        accuracy=accuracy(cm),
        kappa=cohens_kappa(cm),
        confusion=cm,
        score_pairs=len(with_scores),
        pearson=pearson(xs, ys),
        spearman=spearman(xs, ys),
        items=items,
    )


# --- running the judge -----------------------------------------------------------------------


def _judge_spec(scorers: list[dict], name: str | None) -> dict:
    judges = [s for s in scorers if (s.get("type") or s.get("name")) in JUDGE_TYPES]
    if name:
        for s in scorers:
            if name in (s.get("name"), s.get("type")):
                if (s.get("type") or s.get("name")) not in JUDGE_TYPES:
                    raise ValueError(f"Scorer {name!r} is not an LLM judge ({', '.join(JUDGE_TYPES)})")
                return s
        raise ValueError(f"No scorer named {name!r} in the suite")
    if not judges:
        raise ValueError(f"Suite has no LLM judge scorer to calibrate ({', '.join(JUDGE_TYPES)})")
    return judges[0]


def judge_case(case: EvalCase, adapter, judge: Scorer) -> CalibrationItem:
    """Run the system on one labeled case and grade it with the judge. Never raises."""
    label = case.human_label
    assert label is not None
    item = CalibrationItem(case_id=case.id, human_passed=label.passed, human_score=label.score)
    try:
        run = adapter.run(case)
    except Exception as e:
        run = RunResult(case_id=case.id, output="", error=f"adapter crashed: {type(e).__name__}: {e}")
    item.output = run.output
    if run.error:
        item.error = f"system error: {run.error}"
        return item
    try:
        result = judge.score(case, run)
    except Exception as e:
        item.error = f"judge crashed: {type(e).__name__}: {e}"
        return item
    details = result.details or {}
    if details.get("error"):
        item.error = f"judge error: {details['error']}"
    elif details.get("skipped"):
        item.error = f"judge skipped: {details['skipped']}"
    else:
        item.judge_passed = result.passed
        item.judge_score = result.score
        parsed = details.get("parsed")
        if isinstance(parsed, dict) and parsed.get("reason") is not None:
            item.reason = str(parsed["reason"])
    return item


def calibrate_suite(
    suite_path: Path,
    project_root: Path,
    labels_path: Path | None = None,
    adapter_name: str | None = None,
    adapter_options: dict | None = None,
    judge_name: str | None = None,
    judge: Scorer | None = None,
    concurrency: int = 1,
    use_cache: bool = True,
    cache: JudgeCache | None = None,
    on_case_done: Callable[[CalibrationItem], None] | None = None,
) -> CalibrationResult:
    """Grade every human-labeled case in a suite with its LLM judge and measure agreement.

    Labels come from ``labels_path`` (a labels JSONL) and/or inline ``human_label`` fields. The
    judge is the suite scorer named ``judge_name`` or else the first LLM judge scorer; pass a
    ready-made ``judge`` to override it (tests inject fake clients this way).
    """
    suite = load_suite(suite_path, project_root=project_root)
    cases = load_cases(Path(suite.dataset))
    if labels_path is not None:
        cases = apply_labels(cases, load_labels(labels_path))
    labeled = labeled_cases(cases)
    if not labeled:
        raise ValueError("No human-labeled cases: pass --labels or add human_label to the dataset")

    if judge is None:
        if use_cache and cache is None:
            cache = JudgeCache.default(project_root)
        judge = build_scorer(_judge_spec(suite.scorers, judge_name), cache=cache if use_cache else None)
    judge_label = f"{getattr(judge, 'provider', '?')}:{getattr(judge, 'model', '?')}"
    if getattr(judge, "name", None) and judge.name != "llm_judge":
        judge_label = f"{judge.name} ({judge_label})"

    adapter_cfg = dict(suite.adapter)
    suite_adapter = adapter_cfg.pop("type", "mock")
    name = adapter_name or suite_adapter
    if name != suite_adapter:
        adapter_cfg = {}
    if adapter_options:
        adapter_cfg.update(adapter_options)
    if name == "mcp":
        adapter_cfg.setdefault("root", str(project_root))
    adapter = build_adapter(name, adapter_cfg)

    def _one(case: EvalCase) -> CalibrationItem:
        item = judge_case(case, adapter, judge)  # type: ignore[arg-type]
        if on_case_done:
            on_case_done(item)
        return item

    try:
        if concurrency <= 1:
            items = [_one(c) for c in labeled]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                items = list(pool.map(_one, labeled))
    finally:
        adapter.close()

    return summarize(items, suite_name=suite.name, judge=judge_label, unlabeled=len(cases) - len(labeled))
