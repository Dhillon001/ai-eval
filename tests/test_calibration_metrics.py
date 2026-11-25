"""Calibration metrics under the scenarios that matter when trusting a judge.

- perfect agreement (and perfect disagreement);
- chance-level judges, which should score kappa ~0 however good their accuracy looks;
- skewed classes, where accuracy flatters a lazy judge but kappa does not;
- missing labels and missing scores, which must be skipped and counted, not guessed.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest
from fakes import FakeOpenAIByPrompt

from ai_eval.calibrate import (
    CalibrationItem,
    ConfusionMatrix,
    accuracy,
    calibrate_suite,
    cohens_kappa,
    confusion_matrix,
    pearson,
    spearman,
    summarize,
)
from ai_eval.scorers.llm_judge import LLMJudgeScorer


def _items(pairs: list[tuple[bool, bool]], scores: list[tuple[float, float]] | None = None) -> list[CalibrationItem]:
    """Calibration items from ``(judge_passed, human_passed)`` pairs, with optional
    ``(judge_score, human_score)`` pairs."""
    out = []
    for i, (judge, human) in enumerate(pairs):
        js, hs = scores[i] if scores else (None, None)
        out.append(
            CalibrationItem(case_id=str(i), human_passed=human, judge_passed=judge, judge_score=js, human_score=hs)
        )
    return out


def _cm(tp: int, fp: int, fn: int, tn: int) -> ConfusionMatrix:
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


# --- perfect ---------------------------------------------------------------------------------


def test_perfect_agreement():
    pairs = [(True, True)] * 6 + [(False, False)] * 4
    grades = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.0]
    scores = [(g, g) for g in grades]  # judge scores match the human grades exactly
    res = summarize(_items(pairs, scores), suite_name="s", judge="j")
    assert (res.confusion.tp, res.confusion.fp, res.confusion.fn, res.confusion.tn) == (6, 0, 0, 4)
    assert res.accuracy == 1.0
    assert res.kappa == pytest.approx(1.0)
    assert res.disagreements == [] and res.errors == []
    assert res.score_pairs == 10
    assert res.pearson == pytest.approx(1.0) and res.spearman == pytest.approx(1.0)


def test_perfect_disagreement_is_negative_kappa():
    cm = confusion_matrix([(True, False)] * 5 + [(False, True)] * 5)
    assert accuracy(cm) == 0.0
    assert cohens_kappa(cm) == pytest.approx(-1.0)


def test_kappa_is_symmetric_and_bounded():
    rng = random.Random(7)
    for _ in range(200):
        cm = _cm(*(rng.randint(0, 20) for _ in range(4)))
        k = cohens_kappa(cm)
        swapped = _cm(cm.tp, cm.fn, cm.fp, cm.tn)  # swap which rater is "judge"
        if k is None:
            assert cohens_kappa(swapped) is None
        else:
            assert cohens_kappa(swapped) == pytest.approx(k)
            assert -1.0 - 1e-9 <= k <= 1.0 + 1e-9


# --- chance level ----------------------------------------------------------------------------


def test_chance_level_judge_has_zero_kappa():
    # Judge and human independent, both 50/50: agreement is exactly what chance predicts.
    assert cohens_kappa(_cm(25, 25, 25, 25)) == pytest.approx(0.0)
    # Independent with unequal marginals (judge passes 80%, human 30%): p_o == p_e.
    assert cohens_kappa(_cm(24, 56, 6, 14)) == pytest.approx(0.0)


def test_always_pass_judge_is_chance_on_balanced_set():
    cm = confusion_matrix([(True, True)] * 50 + [(True, False)] * 50)
    assert accuracy(cm) == 0.5
    assert cohens_kappa(cm) == pytest.approx(0.0)


def test_random_judge_kappa_near_zero():
    rng = random.Random(1234)
    humans = [rng.random() < 0.5 for _ in range(4000)]
    pairs = [(rng.random() < 0.5, h) for h in humans]
    k = cohens_kappa(confusion_matrix(pairs))
    assert k is not None and abs(k) < 0.05


def test_uncorrelated_scores():
    xs = [1.0, 2.0, 3.0, 4.0]
    # Symmetric "hill" with ties: no monotone relation at all.
    assert spearman(xs, [1.0, 4.0, 4.0, 1.0]) == pytest.approx(0.0)
    assert pearson(xs, [1.0, 4.0, 4.0, 1.0]) == pytest.approx(0.0)


# --- skewed classes --------------------------------------------------------------------------


def test_skewed_set_accuracy_flatters_a_lazy_judge():
    # 90% of humans pass; a judge that passes everything is 90% "accurate" yet no better than chance.
    lazy = confusion_matrix([(True, True)] * 90 + [(True, False)] * 10)
    assert accuracy(lazy) == pytest.approx(0.9)
    assert cohens_kappa(lazy) == pytest.approx(0.0)


def test_skewed_set_kappa_rewards_catching_rare_failures():
    # Same 90% accuracy as the lazy judge, but this one catches 8/10 human failures.
    sharp = _cm(tp=82, fp=2, fn=8, tn=8)
    assert accuracy(sharp) == pytest.approx(0.9)
    # p_o = .9, judge_pass = .84, human_pass = .9 -> p_e = .756 + .016 = .772
    assert cohens_kappa(sharp) == pytest.approx((0.9 - 0.772) / (1 - 0.772))
    assert cohens_kappa(sharp) > 0.5


def test_skewed_toward_failures():
    # Rare passes: missing the one real pass costs a lot of kappa despite 95% accuracy.
    cm = _cm(tp=0, fp=0, fn=1, tn=19)
    assert accuracy(cm) == pytest.approx(0.95)
    assert cohens_kappa(cm) == pytest.approx(0.0)
    assert cohens_kappa(_cm(tp=1, fp=0, fn=0, tn=19)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "pairs",
    [[(True, True)] * 5, [(False, False)] * 5],
    ids=["all-pass", "all-fail"],
)
def test_single_class_kappa_undefined(pairs):
    res = summarize(_items(pairs), suite_name="s", judge="j")
    assert res.accuracy == 1.0
    assert res.kappa is None


# --- missing labels and scores ---------------------------------------------------------------


def test_missing_human_scores_are_left_out_of_correlation():
    items = _items([(True, True), (True, True), (False, False), (False, False)])
    items[0].human_score, items[0].judge_score = 1.0, 0.9
    items[2].human_score, items[2].judge_score = 0.0, 0.1
    items[3].judge_score = 0.2  # judge scored, human didn't
    res = summarize(items, suite_name="s", judge="j")
    assert res.score_pairs == 2
    assert res.pearson == pytest.approx(1.0)
    assert res.n == 4 and res.kappa == pytest.approx(1.0)  # pass/fail metrics still use all four


def test_no_human_scores_means_no_correlation():
    res = summarize(_items([(True, True), (False, False), (True, False)]), suite_name="s", judge="j")
    assert res.score_pairs == 0 and res.pearson is None and res.spearman is None


def test_all_judge_errors_leave_metrics_undefined():
    items = [CalibrationItem(case_id=str(i), human_passed=True, error="judge error: x") for i in range(3)]
    res = summarize(items, suite_name="s", judge="j", unlabeled=2)
    assert (res.labeled, res.n, res.unlabeled) == (3, 0, 2)
    assert res.accuracy is None and res.kappa is None and res.pearson is None
    assert res.confusion.total == 0 and len(res.errors) == 3


def _verdict_judge(prompt: str) -> str:
    m = re.search(r"score=([0-9.]+)", prompt)
    score = float(m.group(1)) if m else (0.9 if "GOOD" in prompt else 0.1)
    return json.dumps({"pass": "GOOD" in prompt, "score": score, "reason": "fake"})


def _suite(tmp_path: Path, rows: list[dict], labels: list[dict] | None = None) -> Path:
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    if labels is not None:
        (tmp_path / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: calib\ndataset: cases.jsonl\nadapter:\n  type: mock\n  template: '{input}'\n"
        "scorers:\n  - type: llm_judge\n",
        encoding="utf-8",
    )
    return suite


def test_calibrate_suite_skips_and_counts_unlabeled_cases(tmp_path):
    rows = [{"id": str(i), "input": "GOOD" if i % 2 else "bad"} for i in range(10)]
    labels = [{"id": "1", "passed": True}, {"id": "2", "passed": False}, {"id": "3", "pass": False}]
    suite = _suite(tmp_path, rows, labels)
    client = FakeOpenAIByPrompt(_verdict_judge)
    res = calibrate_suite(suite, tmp_path, labels_path=tmp_path / "labels.jsonl", judge=LLMJudgeScorer(client=client))
    assert (res.labeled, res.unlabeled, res.n) == (3, 7, 3)
    assert sorted(i.case_id for i in res.items) == ["1", "2", "3"]
    assert len(client.calls) == 3  # unlabeled cases are never sent to the judge
    assert [i.case_id for i in res.disagreements] == ["3"]
    assert res.score_pairs == 0 and res.pearson is None  # labels carried no scores


def test_calibrate_suite_mixes_inline_and_file_labels(tmp_path):
    rows = [
        {"id": "a", "input": "GOOD score=0.9", "human_label": {"pass": False, "score": 0.1}},  # overridden
        {"id": "b", "input": "bad score=0.2", "human_label": {"pass": False, "score": 0.3}},  # inline only
        {"id": "c", "input": "GOOD score=0.6"},  # no label anywhere
    ]
    suite = _suite(tmp_path, rows, [{"id": "a", "passed": True, "score": 0.95}])
    judge = LLMJudgeScorer(client=FakeOpenAIByPrompt(_verdict_judge))
    res = calibrate_suite(suite, tmp_path, labels_path=tmp_path / "labels.jsonl", judge=judge)
    assert (res.labeled, res.unlabeled) == (2, 1)
    assert res.accuracy == 1.0 and res.kappa == pytest.approx(1.0)
    by_id = {i.case_id: i for i in res.items}
    assert by_id["a"].human_passed is True and by_id["a"].human_score == 0.95


def test_calibrate_suite_rejects_labels_for_unknown_cases(tmp_path):
    suite = _suite(tmp_path, [{"id": "1", "input": "x"}], [{"id": "nope", "passed": True}])
    judge = LLMJudgeScorer(client=FakeOpenAIByPrompt(_verdict_judge))
    with pytest.raises(ValueError, match="unknown case ids"):
        calibrate_suite(suite, tmp_path, labels_path=tmp_path / "labels.jsonl", judge=judge)
