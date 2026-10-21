from ai_eval.models import EvalCase, RunResult
from ai_eval.scorers.checklist import ChecklistScorer
from ai_eval.scorers.exact import ExactMatchScorer


def test_exact_match():
    case = EvalCase(id="1", input="x", expected="Hello")
    run = RunResult(case_id="1", output="hello")
    s = ExactMatchScorer().score(case, run)
    assert s.passed and s.score == 1.0


def test_checklist_missing():
    case = EvalCase(id="1", input="x", checks=["refund", "policy"])
    run = RunResult(case_id="1", output="We offer refund")
    s = ChecklistScorer().score(case, run)
    assert not s.passed
    assert s.score == 0.5
