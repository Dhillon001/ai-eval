from pathlib import Path

import pytest

from ai_eval.adapters import build_adapter
from ai_eval.adapters import http as http_mod
from ai_eval.adapters.chat_adapter import AnthropicAdapter, OpenAIAdapter
from ai_eval.adapters.http import HTTPAdapter
from ai_eval.models import EvalCase
from ai_eval.runner import run_suite
from ai_eval.store import RunStore
from fakes import FakeAnthropic, FakeHTTPService, FakeOpenAI


def test_openai_adapter_with_context():
    client = FakeOpenAI("refund within 30 days")
    r = OpenAIAdapter(client=client).run(EvalCase(id="1", input="refund?", context="Policy: 30 days"))
    assert r.output == "refund within 30 days" and r.error is None
    assert (r.prompt_tokens, r.completion_tokens) == (10, 5)
    assert "Policy: 30 days" in client.calls[0]["messages"][1]["content"]


def test_anthropic_adapter():
    client = FakeAnthropic("hi from claude")
    r = AnthropicAdapter(system_prompt="be brief", client=client).run(EvalCase(id="1", input="hi"))
    assert r.output == "hi from claude"
    assert (r.prompt_tokens, r.completion_tokens) == (12, 7)
    assert client.calls[0]["system"] == "be brief" and client.calls[0]["max_tokens"] == 1024


def test_api_error_becomes_result_error():
    r = OpenAIAdapter(client=FakeOpenAI(fail=True)).run(EvalCase(id="1", input="x"))
    assert r.output == "" and "boom" in r.error


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = AnthropicAdapter().run(EvalCase(id="1", input="x"))
    assert r.error and ("ANTHROPIC_API_KEY" in r.error or "Install anthropic" in r.error)


def test_build_adapter_names():
    assert isinstance(build_adapter("claude", {"template": "ignored"}), AnthropicAdapter)
    assert isinstance(build_adapter("openai_compat"), OpenAIAdapter)


def test_build_http_adapter_from_options():
    a = build_adapter("http", {"url": "https://svc.example/a", "response_path": "x", "trace_path": "t"})
    assert isinstance(a, HTTPAdapter)
    assert (a.url, a.response_path, a.trace_path) == ("https://svc.example/a", "x", "t")


def test_build_http_adapter_rejects_unknown_options():
    with pytest.raises(ValueError, match="model"):
        build_adapter("http", {"url": "https://svc.example/a", "model": "gpt"})


def test_http_example_suite_runs_against_fake_service(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    svc = FakeHTTPService(json={"data": {"answer": "see context", "retrieved_ids": ["doc-1"]}})
    monkeypatch.setenv("API_TOKEN", "t0ken")
    real_init = http_mod.HTTPAdapter.__init__

    def _init(self, *args, **kwargs):
        real_init(self, *args, transport=svc.transport, **kwargs)

    monkeypatch.setattr(http_mod.HTTPAdapter, "__init__", _init)
    _, summary, outcomes = run_suite(
        suite_path=root / "configs" / "http_example.yaml",
        project_root=root,
        store=RunStore(tmp_path / "runs.db"),
    )
    assert summary.total == len(outcomes) > 0
    assert all(o.run.error is None for o in outcomes)
    assert outcomes[0].run.trace["retrieved_ids"] == ["doc-1"]
    assert "tool_calls" not in outcomes[0].run.trace
    req = svc.requests[0]
    assert str(req.url) == "http://localhost:8000/answer"
    assert req.headers["Authorization"] == "Bearer t0ken"
