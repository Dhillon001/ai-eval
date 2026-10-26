import json

import yaml
from typer.testing import CliRunner

from ai_eval.cache import CachedResponse, JudgeCache, cache_key
from ai_eval.cli import app
from ai_eval.models import EvalCase, RunResult
from ai_eval.scorers import build_scorer
from ai_eval.scorers.llm_judge import LLMJudgeScorer
from fakes import FakeOpenAI

CASE = EvalCase(id="1", input="q")
RUN = RunResult(case_id="1", output="a")


def test_cache_key_depends_on_every_input():
    base = cache_key("openai", "m", "p", system="s")
    assert base == cache_key("openai", "m", "p", system="s")
    assert len({base, cache_key("anthropic", "m", "p", "s"), cache_key("openai", "m2", "p", "s"),
                cache_key("openai", "m", "p2", "s"), cache_key("openai", "m", "p", "s2")}) == 5


def test_cache_roundtrip_and_corrupt_entry_is_miss(tmp_path):
    cache = JudgeCache(tmp_path)
    assert cache.get("ab" * 32) is None
    cache.put("ab" * 32, CachedResponse(text="hi", prompt_tokens=3))
    assert cache.get("ab" * 32) == CachedResponse(text="hi", prompt_tokens=3)
    (tmp_path / "cd").mkdir()
    (tmp_path / "cd" / f"{'cd' * 32}.json").write_text("{not json")
    assert cache.get("cd" * 32) is None


def test_judge_second_call_is_served_from_cache(tmp_path):
    client = FakeOpenAI('{"pass": true, "score": 0.9}')
    cache = JudgeCache(tmp_path)
    first = LLMJudgeScorer(client=client, cache=cache).score(CASE, RUN)
    second = LLMJudgeScorer(client=client, cache=cache).score(CASE, RUN)
    assert len(client.calls) == 1
    assert first.details["cached"] is False and second.details["cached"] is True
    assert (second.passed, second.score) == (first.passed, first.score)


def test_judge_cache_miss_when_output_or_model_changes(tmp_path):
    client = FakeOpenAI('{"pass": true, "score": 1.0}')
    cache = JudgeCache(tmp_path)
    LLMJudgeScorer(client=client, cache=cache).score(CASE, RUN)
    LLMJudgeScorer(client=client, cache=cache).score(CASE, RunResult(case_id="1", output="different"))
    LLMJudgeScorer(client=client, cache=cache, model="other").score(CASE, RUN)
    assert len(client.calls) == 3


def test_judge_errors_are_not_cached(tmp_path):
    cache = JudgeCache(tmp_path)
    failed = LLMJudgeScorer(client=FakeOpenAI(fail=True), cache=cache, max_retries=0).score(CASE, RUN)
    assert "error" in failed.details
    ok_client = FakeOpenAI('{"pass": true, "score": 1.0}')
    LLMJudgeScorer(client=ok_client, cache=cache).score(CASE, RUN)
    assert len(ok_client.calls) == 1


def test_build_scorer_passes_cache_unless_disabled(tmp_path):
    cache = JudgeCache(tmp_path)
    assert build_scorer({"type": "llm_judge"}, cache=cache).cache is cache
    assert build_scorer({"type": "faithfulness"}, cache=cache).cache is cache
    assert build_scorer({"type": "llm_judge", "cache": False}, cache=cache).cache is None
    assert build_scorer({"type": "exact", "cache": False}, cache=cache).name == "exact"


def _judge_suite(tmp_path):
    (tmp_path / "data.jsonl").write_text(json.dumps({"id": "1", "input": "q", "expected": "a"}) + "\n")
    suite = {
        "name": "judged",
        "dataset": "data.jsonl",
        "adapter": {"type": "mock"},
        "scorers": [{"type": "llm_judge", "provider": "openai"}],
    }
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump(suite))
    return tmp_path / "suite.yaml"


def test_cli_run_uses_cache_and_no_cache_bypasses_it(tmp_path, monkeypatch):
    client = FakeOpenAI('{"pass": true, "score": 1.0}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ai_eval.scorers.llm_judge.make_client", lambda provider: client)
    suite = _judge_suite(tmp_path)
    runner = CliRunner()

    first = runner.invoke(app, ["run", "--suite", str(suite)])
    second = runner.invoke(app, ["run", "--suite", str(suite)])
    assert first.exit_code == 0 and second.exit_code == 0, first.output + second.output
    assert len(client.calls) == 1
    assert "judge cache hits: 1" in second.output
    assert any((tmp_path / ".ai-eval" / "cache").rglob("*.json"))

    third = runner.invoke(app, ["run", "--suite", str(suite), "--no-cache"])
    assert third.exit_code == 0, third.output
    assert len(client.calls) == 2
    assert "judge cache hits" not in third.output
