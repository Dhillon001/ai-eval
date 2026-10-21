from ai_eval.models import EvalCase, RunResult
from ai_eval.scorers import build_scorer
from ai_eval.scorers.llm_judge import LLMJudgeScorer
from fakes import FakeAnthropic, FakeOpenAI


def test_judge_parses_fenced_json_anthropic():
    j = LLMJudgeScorer(provider="anthropic", client=FakeAnthropic('```json\n{"pass": true, "score": 0.9}\n```'))
    s = j.score(EvalCase(id="1", input="q"), RunResult(case_id="1", output="a"))
    assert s.passed and s.score == 0.9 and s.details["judge"] == "anthropic:claude-haiku-5-5"


def test_judge_threshold_when_no_pass_key():
    j = LLMJudgeScorer(client=FakeOpenAI('{"score": 0.5}'), pass_threshold=0.7)
    s = j.score(EvalCase(id="1", input="q"), RunResult(case_id="1", output="a"))
    assert not s.passed and s.score == 0.5


def test_judge_garbage_output_fails_closed():
    j = LLMJudgeScorer(client=FakeOpenAI("I think it's fine"))
    s = j.score(EvalCase(id="1", input="q"), RunResult(case_id="1", output="a"))
    assert not s.passed and s.score == 0.0


def test_faithfulness_includes_context_and_skips_without_it():
    client = FakeOpenAI('{"pass": false, "score": 0.2, "unsupported_claims": ["60 days"]}')
    f = build_scorer({"type": "faithfulness", "client": client})
    assert f.name == "faithfulness"
    s = f.score(EvalCase(id="1", input="q", context="Returns: 30 days"), RunResult(case_id="1", output="60 days"))
    assert not s.passed and "Returns: 30 days" in client.calls[0]["messages"][1]["content"]
    skipped = f.score(EvalCase(id="2", input="q"), RunResult(case_id="2", output="x"))
    assert skipped.passed and skipped.details["skipped"] == "no context"
