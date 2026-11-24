from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_eval.dataset import apply_labels, labeled_cases, load_cases, load_labels
from ai_eval.models import EvalCase, HumanLabel


def _write_jsonl(path: Path, rows: list[object]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_human_label_accepts_pass_alias_and_optional_score():
    assert HumanLabel.model_validate({"pass": True}) == HumanLabel(passed=True)
    label = HumanLabel.model_validate({"passed": False, "score": 0.25, "notes": "misses refund"})
    assert (label.passed, label.score, label.notes) == (False, 0.25, "misses refund")


@pytest.mark.parametrize(
    "data",
    [
        {},  # passed is required
        {"passed": True, "score": 1.5},
        {"passed": True, "score": -0.1},
        {"passed": True, "verdict": "good"},  # unknown keys are rejected (catches typos)
    ],
)
def test_human_label_validation_errors(data):
    with pytest.raises(ValidationError):
        HumanLabel.model_validate(data)


def test_inline_human_label_loads_from_dataset(tmp_path):
    path = _write_jsonl(
        tmp_path / "cases.jsonl",
        [
            {"id": "a", "input": "hi", "human_label": {"pass": True, "score": 0.9}},
            {"id": "b", "input": "yo"},
        ],
    )
    cases = load_cases(path)
    assert cases[0].human_label == HumanLabel(passed=True, score=0.9)
    assert cases[1].human_label is None
    assert [c.id for c in labeled_cases(cases)] == ["a"]


def test_invalid_inline_label_reports_line(tmp_path):
    path = _write_jsonl(
        tmp_path / "cases.jsonl",
        [{"id": "a", "input": "hi"}, {"id": "b", "input": "x", "human_label": {"score": 0.5}}],
    )
    with pytest.raises(ValueError, match=r"cases.jsonl:2: invalid case"):
        load_cases(path)


def test_single_turn_cases_unchanged_without_labels():
    case = EvalCase(id="a", input="hi")
    assert case.human_label is None
    assert EvalCase.model_validate(case.model_dump()) == case


def test_load_labels_file(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "# human labels\n"
        '{"id": "a", "pass": true, "score": 1.0}\n'
        "\n"
        '{"case_id": "b", "passed": false, "notes": "hallucinated a date"}\n',
        encoding="utf-8",
    )
    labels = load_labels(path)
    assert labels == {
        "a": HumanLabel(passed=True, score=1.0),
        "b": HumanLabel(passed=False, notes="hallucinated a date"),
    }


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([{"passed": True}], r":1: invalid label: missing case 'id'"),
        ([{"id": "a"}], r":1: invalid label"),
        ([{"id": "a", "passed": True, "score": 2}], r":1: invalid label"),
        ([{"id": "a", "case_id": "a", "passed": True}], r"either 'id' or 'case_id'"),
        ([["a", True]], r"expected a JSON object"),
        (
            [{"id": "a", "passed": True}, {"id": "a", "passed": False}],
            r":2: duplicate label for case 'a'",
        ),
    ],
)
def test_load_labels_validation(tmp_path, rows, match):
    path = _write_jsonl(tmp_path / "labels.jsonl", rows)
    with pytest.raises(ValueError, match=match):
        load_labels(path)


def test_load_labels_rejects_bad_json_and_empty_files(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "passed": true\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:1: invalid label"):
        load_labels(bad)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("# nothing yet\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No labels found"):
        load_labels(empty)


def test_apply_labels_merges_and_file_overrides_inline():
    cases = [
        EvalCase(id="a", input="1", human_label=HumanLabel(passed=True)),
        EvalCase(id="b", input="2"),
        EvalCase(id="c", input="3", human_label=HumanLabel(passed=False, score=0.1)),
    ]
    merged = apply_labels(cases, {"a": HumanLabel(passed=False), "b": HumanLabel(passed=True)})

    assert [c.human_label for c in merged] == [
        HumanLabel(passed=False),
        HumanLabel(passed=True),
        HumanLabel(passed=False, score=0.1),
    ]
    # Inputs are not mutated.
    assert cases[0].human_label == HumanLabel(passed=True)
    assert cases[1].human_label is None
    assert len(labeled_cases(merged)) == 3


def test_apply_labels_rejects_unknown_case_ids():
    cases = [EvalCase(id="a", input="1")]
    labels = {cid: HumanLabel(passed=True) for cid in ["a", "zz", "b1", "b2", "b3", "b4", "b5"]}
    with pytest.raises(ValueError, match=r"unknown case ids: 'b1', .* \(\+1 more\)"):
        apply_labels(cases, labels)
