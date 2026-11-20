"""Agent loop over OpenAI function calling, driven by a scripted fake OpenAI client
(no network, no API keys), plus provider dispatch and the ``mcp`` adapter with ``provider: openai``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import (
    FakeAnthropicAgent,
    FakeAPIError,
    FakeOpenAIAgent,
    FakeToolExecutor,
    openai_tool_call,
    tool_use_block,
)

from ai_eval.adapters import build_adapter
from ai_eval.adapters.mcp import MCPAdapter
from ai_eval.agent_loop import (
    ToolPolicy,
    anthropic_tool_specs,
    openai_tool_specs,
    run_agent,
    run_anthropic_agent,
    run_openai_agent,
)
from ai_eval.mcp_client import MCPTool
from ai_eval.models import EvalCase

MODEL = "gpt-test"
FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"

STOCK = {"BRK-001": 12, "FLT-204": 0}


def _lookup(sku: str):
    if sku not in STOCK:
        return (f"unknown sku: {sku}", True)
    return f'{{"sku": "{sku}", "stock": {STOCK[sku]}}}'


def _inventory() -> FakeToolExecutor:
    return FakeToolExecutor(
        {"lookup": _lookup, "add": lambda a, b: str(a + b), "delete_part": lambda sku: "deleted"},
        descriptions={"lookup": "Look up a part by SKU."},
        schemas={"lookup": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}},
    )


def _run(client, tools, **kw):
    kw.setdefault("sleep", lambda _s: None)
    return run_openai_agent(client, tools, model=MODEL, user=kw.pop("user", "How many BRK-001?"), **kw)


def _tool_messages(call: dict) -> list[dict]:
    """The trailing ``role: tool`` messages of a recorded ``chat.completions.create`` call."""
    out = []
    for m in reversed(call["messages"]):
        if m["role"] != "tool":
            break
        out.insert(0, m)
    return out


# -- tool definitions ------------------------------------------------------------------------


def test_openai_tool_specs_wrap_mcp_tools_as_functions():
    tools = [
        MCPTool(name="lookup", description="Find a part.", input_schema={"properties": {"sku": {"type": "string"}}}),
        MCPTool(name="ping"),
    ]
    assert openai_tool_specs(tools) == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find a part.",
                "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}},
            },
        },
        {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}},
    ]
    # Same schemas as the Anthropic specs, just a different envelope.
    assert [s["function"]["parameters"] for s in openai_tool_specs(tools)] == [
        s["input_schema"] for s in anthropic_tool_specs(tools)
    ]


def test_request_shape_system_message_and_tools():
    client = FakeOpenAIAgent(["Hello."])
    run = _run(client, _inventory(), system="be brief")
    [call] = client.calls
    assert call["model"] == MODEL and call["temperature"] == 0.0
    assert call["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "How many BRK-001?"},
    ]
    assert [t["function"]["name"] for t in call["tools"]] == ["lookup", "add", "delete_part"]
    assert "max_completion_tokens" not in call and "max_tokens" not in call
    assert run.provider == "openai" and run.output == "Hello." and run.stop_reason == "final_answer"


def test_max_tokens_is_sent_as_max_completion_tokens_when_set():
    client = FakeOpenAIAgent(["ok"])
    _run(client, _inventory(), max_tokens=256)
    assert client.calls[0]["max_completion_tokens"] == 256


def test_no_tools_omits_tools_kwarg():
    client = FakeOpenAIAgent(["Hello."])
    _run(client, FakeToolExecutor({}))
    assert "tools" not in client.calls[0]


# -- single and multi-step -------------------------------------------------------------------


def test_single_tool_call_then_final_answer():
    client = FakeOpenAIAgent(
        [
            ("Let me check.", [openai_tool_call("lookup", {"sku": "BRK-001"}, id="call_1")]),
            "There are 12 BRK-001 brake pads in stock.",
        ]
    )
    tools = _inventory()
    run = _run(client, tools)

    assert run.stop_reason == "final_answer" and run.error is None
    assert run.output == "There are 12 BRK-001 brake pads in stock."
    assert run.steps == 2
    assert tools.calls == [("lookup", {"sku": "BRK-001"})]
    [rec] = run.tool_calls
    assert (rec.name, rec.args, rec.error, rec.step, rec.id) == ("lookup", {"sku": "BRK-001"}, None, 1, "call_1")

    msgs = client.calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[2] == {
        "role": "assistant",
        "content": "Let me check.",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"sku": "BRK-001"}'}}
        ],
    }
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"sku": "BRK-001", "stock": 12}'}


def test_multi_step_and_parallel_calls():
    client = FakeOpenAIAgent(
        [
            [openai_tool_call("lookup", {"sku": "BRK-001"}, id="a"), openai_tool_call("lookup", {"sku": "FLT-204"}, id="b")],
            [openai_tool_call("add", {"a": 12, "b": 0}, id="c")],
            "12 in total.",
        ],
        prompt_tokens=30,
        completion_tokens=4,
    )
    run = _run(client, _inventory())

    assert run.output == "12 in total." and run.steps == 3
    assert [(c.name, c.step, c.id) for c in run.tool_calls] == [("lookup", 1, "a"), ("lookup", 1, "b"), ("add", 2, "c")]
    assert [m["tool_call_id"] for m in _tool_messages(client.calls[1])] == ["a", "b"]
    assert client.calls[1]["messages"][2]["content"] is None  # tool-only turn replays content=None
    assert [m["tool_call_id"] for m in _tool_messages(client.calls[2])] == ["c"]
    assert run.prompt_tokens == 90 and run.completion_tokens == 12


# -- errors ------------------------------------------------------------------------------------


def test_tool_error_is_recorded_and_sent_back():
    client = FakeOpenAIAgent([[openai_tool_call("lookup", {"sku": "NOPE"}, id="x")], "That SKU does not exist."])
    run = _run(client, _inventory())
    [rec] = run.tool_calls
    assert rec.error == "unknown sku: NOPE" and rec.result == "unknown sku: NOPE"
    assert _tool_messages(client.calls[1]) == [{"role": "tool", "tool_call_id": "x", "content": "unknown sku: NOPE"}]
    assert run.output == "That SKU does not exist." and run.error is None


def test_tool_exception_is_reported_back_not_raised():
    def broken(**_kw):
        raise TimeoutError("took too long")

    client = FakeOpenAIAgent([[openai_tool_call("slow", {}, id="s")], "Sorry."])
    run = _run(client, FakeToolExecutor({"slow": broken}))
    assert run.tool_calls[0].error == "TimeoutError: took too long"
    assert _tool_messages(client.calls[1])[0]["content"] == "Error: TimeoutError: took too long"


@pytest.mark.parametrize(
    ("raw", "message"),
    [("{not json", "invalid JSON arguments"), ("[1, 2]", "arguments must be an object")],
)
def test_malformed_arguments_are_not_executed(raw, message):
    client = FakeOpenAIAgent([[openai_tool_call("lookup", raw, id="m")], "Sorry."])
    tools = _inventory()
    run = _run(client, tools)
    assert tools.calls == []
    assert message in run.tool_calls[0].error
    # The raw string is replayed unchanged so the conversation stays consistent.
    assert client.calls[1]["messages"][2]["tool_calls"][0]["function"]["arguments"] == raw
    assert _tool_messages(client.calls[1])[0]["content"].startswith("Error: ")


def test_empty_arguments_string_means_no_args():
    client = FakeOpenAIAgent([[openai_tool_call("ping", "", id="p")], "pong"])
    tools = FakeToolExecutor({"ping": lambda: ""})
    run = _run(client, tools)
    assert tools.calls == [("ping", {})]
    assert run.tool_calls[0].error is None
    assert _tool_messages(client.calls[1])[0]["content"] == "(no output)"


def test_max_steps_hit():
    client = FakeOpenAIAgent([("still looking", [openai_tool_call("lookup", {"sku": "BRK-001"})])], repeat_last=True)
    run = _run(client, _inventory(), max_steps=3)
    assert run.stop_reason == "max_steps" and run.steps == 3 and len(client.calls) == 3
    assert run.output == "still looking"
    assert "max_steps (3)" in run.error
    assert len(run.tool_calls) == 3


def test_max_steps_must_be_positive():
    with pytest.raises(ValueError, match="max_steps"):
        _run(FakeOpenAIAgent([]), _inventory(), max_steps=0)


def test_transient_provider_errors_are_retried():
    client = FakeOpenAIAgent([FakeAPIError(503, retry_after=0), "done"])
    run = _run(client, _inventory())
    assert run.output == "done" and run.retries == 1 and run.stop_reason == "final_answer"


def test_provider_error_ends_run_with_last_text():
    client = FakeOpenAIAgent([("checking", [openai_tool_call("lookup", {"sku": "BRK-001"})]), ValueError("bad request")])
    run = _run(client, _inventory(), max_retries=0)
    assert run.stop_reason == "error" and run.output == "checking"
    assert run.error == "ValueError: bad request"
    assert len(run.tool_calls) == 1


# -- tool policy -------------------------------------------------------------------------------


def test_policy_hides_and_blocks_tools():
    client = FakeOpenAIAgent([[openai_tool_call("delete_part", {"sku": "BRK-001"}, id="d")], "I can't do that."])
    tools = _inventory()
    run = _run(client, tools, policy=ToolPolicy(denied_tools=["delete_*"], tool_timeout=2.5))
    assert [t["function"]["name"] for t in client.calls[0]["tools"]] == ["lookup", "add"]
    assert tools.calls == []
    [rec] = run.tool_calls
    assert rec.blocked and "denied" in rec.error
    assert rec.to_trace()["blocked"] is True
    assert _tool_messages(client.calls[1])[0]["content"].startswith("Error: tool 'delete_part' is denied")


def test_policy_timeout_is_passed_to_executor():
    client = FakeOpenAIAgent([[openai_tool_call("lookup", {"sku": "BRK-001"})], "12"])
    tools = _inventory()
    _run(client, tools, policy=ToolPolicy(tool_timeout=2.5))
    assert tools.timeouts == [2.5]


# -- provider parity and dispatch ---------------------------------------------------------------


def test_trace_matches_anthropic_loop():
    """The same conversation produces the same tool_calls trace on both providers."""
    oa = _run(
        FakeOpenAIAgent([[openai_tool_call("lookup", {"sku": "BRK-001"}, id="t1")], "12."]), _inventory()
    ).to_run_result("c")
    an = run_anthropic_agent(
        FakeAnthropicAgent([[tool_use_block("lookup", {"sku": "BRK-001"}, id="t1")], "12."]),
        _inventory(),
        model=MODEL,
        user="How many BRK-001?",
    ).to_run_result("c")
    assert oa.trace["tool_calls"] == an.trace["tool_calls"]
    assert oa.output == an.output
    assert (oa.trace["provider"], an.trace["provider"]) == ("openai", "anthropic")


def test_run_agent_dispatches_by_provider():
    run = run_agent("openai", FakeOpenAIAgent(["hi"]), _inventory(), model=MODEL, user="q")
    assert run.provider == "openai"
    run = run_agent("anthropic", FakeAnthropicAgent(["hi"]), _inventory(), model=MODEL, user="q")
    assert run.provider == "anthropic"
    with pytest.raises(ValueError, match="provider"):
        run_agent("gemini", None, _inventory(), model=MODEL, user="q")


# -- mcp adapter with provider: openai ----------------------------------------------------------


def test_mcp_adapter_openai_provider():
    tools = FakeToolExecutor({"get_part": lambda sku: json.dumps({"sku": sku, "price_usd": 9.75})})
    client = FakeOpenAIAgent([[openai_tool_call("get_part", {"sku": "FLT-204"})], "It costs $9.75."])
    adapter = build_adapter("mcp", {"provider": "openai", "server": {"command": "python"}})
    assert adapter.model == "gpt-4o-mini"  # provider-specific default model
    adapter = MCPAdapter(provider="openai", client=client, tools=tools, system_prompt="be brief")

    result = adapter.run(EvalCase(id="price", input="Price of FLT-204?"))

    assert result.error is None and result.output == "It costs $9.75."
    assert result.trace["adapter"] == "mcp" and result.trace["provider"] == "openai"
    assert result.trace["model"] == "gpt-4o-mini"
    assert [c["name"] for c in result.trace["tool_calls"]] == ["get_part"]
    assert client.calls[0]["messages"][0] == {"role": "system", "content": "be brief"}
    assert "max_completion_tokens" not in client.calls[0]


def test_mcp_adapter_anthropic_keeps_default_max_tokens():
    client = FakeAnthropicAgent(["ok"])
    MCPAdapter(client=client, tools=FakeToolExecutor({})).run(EvalCase(id="a", input="q"))
    assert client.calls[0]["max_tokens"] == 1024


def test_openai_agent_against_fake_mcp_server():
    pytest.importorskip("mcp")
    from ai_eval.mcp_client import MCPClient, MCPServerConfig

    client = FakeOpenAIAgent(
        [
            [openai_tool_call("lookup", {"sku": "BRK-001"}, id="l1"), openai_tool_call("lookup", {"sku": "XXX"}, id="l2")],
            [openai_tool_call("add", {"a": 12, "b": 3}, id="a1")],
            "15.",
        ]
    )
    config = MCPServerConfig(command="python", args=[str(FAKE_SERVER)], startup_timeout=20, call_timeout=5)
    with MCPClient(config) as tools:
        run = _run(client, tools)

    assert run.stop_reason == "final_answer" and run.output == "15."
    assert {t["function"]["name"] for t in client.calls[0]["tools"]} >= {"echo", "add", "lookup"}
    ok, missing, added = run.tool_calls
    assert ok.error is None and "brake pad" in ok.result
    assert missing.error and "lookup" in missing.error
    assert added.error is None and "15" in added.result
