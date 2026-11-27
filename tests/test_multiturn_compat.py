"""Multi-turn adapter edge cases and backward compatibility for single-turn cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import FakeAnthropic, FakeAPIError, FakeHTTPService, FakeOpenAI

from ai_eval.adapters import build_adapter
from ai_eval.adapters.chat_adapter import AnthropicAdapter, OpenAIAdapter, split_history
from ai_eval.adapters.http import HTTPAdapter, render_template
from ai_eval.adapters.mock import MockAdapter
from ai_eval.llm import LLMError, chat
from ai_eval.models import EvalCase
from ai_eval.runner import run_suite
from ai_eval.store import RunStore

ROOT = Path(__file__).resolve().parents[1]

TURNS = [
    {"role": "user", "content": "How much is the pro plan?"},
    {"role": "assistant", "content": "$20 per month."},
    {"role": "user", "content": "Is there a yearly option?"},
    {"role": "assistant", "content": "Yes, $200 per year."},
]


def _multi(**kw) -> EvalCase:
    return EvalCase(id="m", input="And in euros?", messages=TURNS, **kw)


# --- llm.chat -----------------------------------------------------------------------------


def test_chat_rejects_system_role_in_history_before_calling_provider():
    client = FakeOpenAI()
    with pytest.raises(LLMError, match="history roles"):
        chat(client, "openai", "m", "sys", "hi", history=[{"role": "system", "content": "x"}])
    assert client.calls == []


def test_chat_empty_history_matches_no_history():
    a, b = FakeOpenAI(), FakeOpenAI()
    chat(a, "openai", "m", "sys", "hi")
    chat(b, "openai", "m", "sys", "hi", history=[])
    assert a.calls == b.calls


def test_chat_does_not_forward_extra_history_keys():
    client = FakeAnthropic()
    chat(client, "anthropic", "m", "sys", "hi", history=[{**TURNS[0], "name": "x"}, TURNS[1]])
    assert client.calls[0]["messages"][0] == TURNS[0]


def test_chat_resends_identical_history_on_retry():
    client = FakeOpenAI(errors=[FakeAPIError(503, retry_after=0)])
    resp = chat(client, "openai", "m", "sys", "q", history=TURNS, sleep=lambda _s: None)
    assert resp.retries == 1
    assert len(client.calls) == 2
    assert client.calls[0]["messages"] == client.calls[1]["messages"]
    assert client.calls[0]["messages"][1:5] == TURNS


# --- openai / anthropic adapters ----------------------------------------------------------


@pytest.mark.parametrize("adapter_cls,fake", [(OpenAIAdapter, FakeOpenAI), (AnthropicAdapter, FakeAnthropic)])
def test_chat_adapters_send_multiple_prior_turns_in_order(adapter_cls, fake):
    client = fake()
    r = adapter_cls(client=client).run(_multi())
    sent = client.calls[0]["messages"]
    if adapter_cls is OpenAIAdapter:
        sent = sent[1:]  # drop the system message
    assert sent == [*TURNS, {"role": "user", "content": "And in euros?"}]
    assert r.trace["history"] == TURNS
    assert r.trace["user"] == "And in euros?"


def test_context_is_only_attached_to_the_final_user_turn():
    client = FakeOpenAI()
    OpenAIAdapter(client=client).run(_multi(context="EUR rate: 0.9"))
    sent = client.calls[0]["messages"]
    assert sent[1:5] == TURNS
    assert sent[-1]["content"] == "Context:\nEUR rate: 0.9\n\nQuestion:\nAnd in euros?"


def test_system_only_history_merges_into_system_prompt_without_history():
    client = FakeAnthropic()
    case = EvalCase(id="s", input="hi", messages=[{"role": "system", "content": "Be brief."}])
    r = AnthropicAdapter(system_prompt="Base.", client=client).run(case)
    assert client.calls[0]["system"] == "Base.\n\nBe brief."
    assert client.calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert "history" not in r.trace


def test_multiple_case_system_messages_are_joined_in_order():
    case = EvalCase(
        id="s",
        input="hi",
        messages=[{"role": "system", "content": "One."}, {"role": "system", "content": "Two."}, *TURNS[:2]],
    )
    systems, turns = split_history(case)
    assert systems == ["One.", "Two."] and turns == TURNS[:2]
    client = FakeOpenAI()
    OpenAIAdapter(system_prompt="Base.", client=client).run(case)
    assert client.calls[0]["messages"][0] == {"role": "system", "content": "Base.\n\nOne.\n\nTwo."}


def test_single_turn_anthropic_request_unchanged():
    client = FakeAnthropic(text="hello")
    r = AnthropicAdapter(system_prompt="Be helpful.", client=client).run(EvalCase(id="s", input="hi"))
    call = client.calls[0]
    assert call["system"] == "Be helpful."
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert r.output == "hello" and "history" not in r.trace


def test_adapter_does_not_mutate_case_messages():
    case = _multi()
    before = case.model_dump()
    OpenAIAdapter(client=FakeOpenAI()).run(case)
    MockAdapter(template="{history}").run(case)
    HTTPAdapter("https://svc.example/a", transport=FakeHTTPService(json={"output": "ok"}).transport).run(case)
    assert case.model_dump() == before


def test_multi_turn_provider_error_is_reported_not_raised():
    r = OpenAIAdapter(client=FakeOpenAI(fail=True), max_retries=0).run(_multi())
    assert r.output == "" and "boom" in r.error


# --- mock adapter -------------------------------------------------------------------------


def test_mock_default_template_ignores_history():
    assert MockAdapter().run(_multi()).output == "echo:And in euros?"


def test_mock_history_placeholder_is_empty_for_single_turn():
    assert MockAdapter(template="[{history}]{input}").run(EvalCase(id="s", input="hi")).output == "[]hi"


def test_mock_stored_output_replayed_for_multi_turn_case():
    r = MockAdapter(template="{history}").run(_multi(metadata={"mock_output": "€18 per month."}))
    assert r.output == "€18 per month."
    assert r.trace["messages"] == TURNS


def test_build_adapter_mock_template_option_supports_history():
    out = build_adapter("mock", {"template": "{history}"}).run(_multi()).output
    assert out.splitlines() == [f"{t['role']}: {t['content']}" for t in TURNS]


# --- http adapter -------------------------------------------------------------------------


def test_http_placeholders_for_single_turn_case():
    case = EvalCase(id="s", input="hi")
    rendered = render_template({"m": "{messages}", "c": "{conversation}", "t": "n={messages}"}, case)
    assert rendered == {"m": [], "c": [{"role": "user", "content": "hi"}], "t": "n=[]"}


def test_http_custom_template_without_history_placeholders_is_unchanged_for_multi_turn():
    svc = FakeHTTPService(json={"output": "ok"})
    HTTPAdapter("https://svc.example/a", body_template={"q": "{input}"}, transport=svc.transport).run(_multi())
    assert json.loads(svc.requests[0].content) == {"q": "And in euros?"}


def test_http_trace_request_records_multi_turn_body():
    svc = FakeHTTPService(json={"output": "ok"})
    r = HTTPAdapter("https://svc.example/a", transport=svc.transport).run(_multi())
    assert r.trace["request"]["messages"] == TURNS
    assert r.output == "ok"


# --- end to end ---------------------------------------------------------------------------


def test_run_suite_mixes_single_and_multi_turn_cases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [
        {"id": "single", "input": "hi", "expected": "echo:hi"},
        {"id": "multi", "input": "And in euros?", "messages": TURNS, "expected": "echo:And in euros?"},
    ]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "suite.yaml").write_text(
        "name: mixed\ndataset: cases.jsonl\nadapter:\n  type: mock\nscorers:\n  - type: exact\n"
    )
    store = RunStore(tmp_path / "runs.db")
    run_id, summary, outcomes = run_suite(
        suite_path=tmp_path / "suite.yaml", project_root=tmp_path, store=store
    )
    assert summary.total == 2 and summary.pass_rate == 1.0
    by_id = {o.case_id: o for o in outcomes}
    assert by_id["multi"].run.trace["messages"] == TURNS
    assert "messages" not in by_id["single"].run.trace
    stored = {o.case_id: o for o in store.get_outcomes(run_id)}
    assert stored["multi"].run.trace["messages"] == TURNS


def test_existing_example_suite_results_unchanged(tmp_path, monkeypatch):
    """Single-turn datasets written before multi-turn support load and run exactly as before."""
    monkeypatch.chdir(ROOT)
    _, summary, outcomes = run_suite(
        suite_path=ROOT / "configs" / "example_suite.yaml",
        project_root=ROOT,
        adapter_name="mock",
        store=RunStore(tmp_path / "runs.db"),
    )
    assert summary.total == 3 and summary.pass_rate == 1.0
    for o in outcomes:
        assert o.run.trace == {"adapter": "mock"}
        assert o.run.output.startswith("Support bot: We can help with ")
