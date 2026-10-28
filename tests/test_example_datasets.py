from collections import Counter
from pathlib import Path

from ai_eval.dataset import load_cases

ROOT = Path(__file__).resolve().parents[1]
RAG_CATEGORIES = {"answerable", "abstain", "conflicting", "multi-hop"}


def test_rag_dataset_is_balanced_and_tagged():
    cases = load_cases(ROOT / "datasets/examples/rag_v1.jsonl")
    assert len(cases) >= 30
    assert len({c.id for c in cases}) == len(cases), "case ids must be unique"

    per_category: Counter[str] = Counter()
    for case in cases:
        tags = case.metadata.get("tags", [])
        assert "rag" in tags, case.id
        categories = RAG_CATEGORIES.intersection(tags)
        assert len(categories) == 1, f"{case.id}: expected exactly one of {sorted(RAG_CATEGORIES)}"
        per_category.update(categories)
        # Every case must be gradable: deterministic checks or a judge rubric (or both).
        assert case.checks or case.metadata.get("judge_criteria"), case.id

    assert set(per_category) == RAG_CATEGORIES
    assert min(per_category.values()) >= 5, per_category


def test_rag_dataset_abstain_cases_have_rubrics():
    cases = load_cases(ROOT / "datasets/examples/rag_v1.jsonl")
    for case in cases:
        if "abstain" in case.metadata["tags"]:
            assert case.metadata.get("judge_criteria"), case.id
