from __future__ import annotations

import json

from ai_eval.adapters.chat_adapter import AnthropicAdapter, OpenAIAdapter
from ai_eval.adapters.http import HTTPAdapter
from ai_eval.adapters.mock import MockAdapter
from ai_eval.models import EvalCase
from fakes import FakeAnthropic, FakeHTTPService, FakeOpenAI

HISTORY = [
    {"role": "system", "content": "Answer in one sentence."},
    {"role": "user", "content": "I ordered a lamp."},
    {"role": "assistant", "content": "Thanks, how can I help with the lamp?"},
]


def _case(**kw) -> EvalCase:
    return EvalCase(id="c1", input="Can I return it?", messages=HISTORY, **kw)


def test_openai_adapter_sends_history_between_system_and_input():
    client = FakeOpenAI(text="Yes, within 30 days.")
    r = OpenAIAdapter(system_prompt="Be helpful.", client=client).run(_case())
    assert r.error is None and r.output == "Yes, within 30 days."
    assert client.calls[0]["messages"] == [
        {"role": "system", "content": "Be helpful.\n\nAnswer in one sentence."},
        {"role": "user", "content": "I ordered a lamp."},
        {"role": "assistant", "content": "Thanks, how can I help with the lamp?"},
        {"role": "user", "content": "Can I return it?"},
    ]
    assert r.trace["history"] == HISTORY[1:]


def test_anthropic_adapter_sends_history_and_merges_case_system():
    client = FakeAnthropic(text="Yes.")
    r = AnthropicAdapter(system_prompt="Be helpful.", client=client).run(_case(context="Policy: 30 days"))
    call = client.calls[0]
    assert call["system"] == "Be helpful.\n\nAnswer in one sentence."
    assert [m["role"] for m in call["messages"]] == ["user", "assistant", "user"]
    assert call["messages"][-1]["content"].startswith("Context:\nPolicy: 30 days")
    assert r.output == "Yes."


def test_single_turn_chat_request_unchanged():
    client = FakeOpenAI()
    r = OpenAIAdapter(system_prompt="Be helpful.", client=client).run(EvalCase(id="s", input="hi"))
    assert client.calls[0]["messages"] == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "hi"},
    ]
    assert "history" not in r.trace


def test_mock_adapter_records_history_and_renders_placeholder():
    r = MockAdapter(template="{history}\nuser: {input}").run(_case())
    assert r.trace["messages"] == HISTORY
    assert r.output.splitlines()[-2:] == ["assistant: Thanks, how can I help with the lamp?", "user: Can I return it?"]
    assert "messages" not in MockAdapter().run(EvalCase(id="s", input="hi")).trace


def test_http_default_body_adds_messages_only_for_multi_turn():
    svc = FakeHTTPService(json={"output": "ok"})
    adapter = HTTPAdapter("https://svc.example/a", transport=svc.transport)
    adapter.run(EvalCase(id="s", input="hi"))
    adapter.run(_case())
    assert json.loads(svc.requests[0].content) == {"input": "hi", "context": None}
    assert json.loads(svc.requests[1].content) == {"input": "Can I return it?", "context": None, "messages": HISTORY}


def test_http_conversation_placeholder():
    svc = FakeHTTPService(json={"output": "ok"})
    HTTPAdapter(
        "https://svc.example/a", body_template={"msgs": "{conversation}", "n": "turns={messages}"}, transport=svc.transport
    ).run(_case())
    body = json.loads(svc.requests[0].content)
    assert body["msgs"] == [*HISTORY, {"role": "user", "content": "Can I return it?"}]
    assert json.loads(body["n"].removeprefix("turns=")) == HISTORY
