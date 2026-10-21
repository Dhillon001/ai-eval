from ai_eval.adapters import build_adapter
from ai_eval.adapters.chat_adapter import AnthropicAdapter, OpenAIAdapter
from ai_eval.models import EvalCase
from fakes import FakeAnthropic, FakeOpenAI


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
