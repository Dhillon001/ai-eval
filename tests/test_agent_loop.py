"""Agent loop driven by a scripted fake Anthropic client (no network, no API keys)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import (
    FakeAnthropicAgent,
    FakeAPIError,
    FakeToolExecutor,
    text_block,
    tool_use_block,
)

from ai_eval.agent_loop import anthropic_tool_specs, run_anthropic_agent
from ai_eval.mcp_client import MCPTool

MODEL = "claude-test"
FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"

STOCK = {"BRK-001": 12, "FLT-204": 0}


def _lookup(sku: str):
    if sku not in STOCK:
        return (f"unknown sku: {sku}", True)
    return f'{{"sku": "{sku}", "stock": {STOCK[sku]}}}'


def _inventory() -> FakeToolExecutor:
    return FakeToolExecutor(
        {"lookup": _lookup, "add": lambda a, b: str(a + b)},
        descriptions={"lookup": "Look up a part by SKU."},
        schemas={"lookup": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}},
    )


def _run(client, tools, **kw):
    kw.setdefault("sleep", lambda _s: None)
    return run_anthropic_agent(client, tools, model=MODEL, user=kw.pop("user", "How many BRK-001?"), **kw)


def _tool_results(call: dict) -> list[dict]:
    """The tool_result blocks in the last user message of a recorded ``messages.create`` call."""
    last = call["messages"][-1]
    assert last["role"] == "user"
    return [b for b in last["content"] if b["type"] == "tool_result"]


# -- tool definitions ------------------------------------------------------------------------


def test_anthropic_tool_specs_maps_mcp_tools():
    specs = anthropic_tool_specs(
        [
            MCPTool(name="lookup", description="Find a part.", input_schema={"properties": {"sku": {"type": "string"}}}),
            MCPTool(name="ping"),
        ]
    )
    assert specs[0] == {
        "name": "lookup",
        "description": "Find a part.",
        "input_schema": {"type": "object", "properties": {"sku": {"type": "string"}}},
    }
    assert specs[1] == {"name": "ping", "input_schema": {"type": "object", "properties": {}}}


def test_tools_are_sent_with_every_request():
    client = FakeAnthropicAgent([[tool_use_block("lookup", {"sku": "BRK-001"})], "12 in stock."])
    _run(client, _inventory())
    for call in client.calls:
        assert call["model"] == MODEL
        assert [t["name"] for t in call["tools"]] == ["lookup", "add"]
    assert client.calls[0]["messages"] == [{"role": "user", "content": "How many BRK-001?"}]


def test_no_tools_omits_tools_kwarg():
    client = FakeAnthropicAgent(["Hello."])
    run = _run(client, FakeToolExecutor({}))
    assert "tools" not in client.calls[0]
    assert run.output == "Hello." and run.tool_calls == []


# -- single call -----------------------------------------------------------------------------


def test_single_tool_call_then_final_answer():
    client = FakeAnthropicAgent(
        [
            [text_block("Let me check."), tool_use_block("lookup", {"sku": "BRK-001"}, id="toolu_1")],
            "There are 12 BRK-001 brake pads in stock.",
        ]
    )
    tools = _inventory()
    run = _run(client, tools)

    assert run.stop_reason == "final_answer" and run.error is None
    assert run.output == "There are 12 BRK-001 brake pads in stock."
    assert run.steps == 2 and len(client.calls) == 2
    assert tools.calls == [("lookup", {"sku": "BRK-001"})]
    [rec] = run.tool_calls
    assert (rec.name, rec.args, rec.error, rec.step, rec.id) == ("lookup", {"sku": "BRK-001"}, None, 1, "toolu_1")
    assert rec.result == '{"sku": "BRK-001", "stock": 12}'

    # The second request replays the assistant turn and answers the tool_use by id.
    msgs = client.calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"sku": "BRK-001"}},
    ]
    assert _tool_results(client.calls[1]) == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"sku": "BRK-001", "stock": 12}'}
    ]


def test_usage_is_summed_across_turns():
    client = FakeAnthropicAgent([[tool_use_block("lookup", {"sku": "BRK-001"})], "12."], input_tokens=30, output_tokens=4)
    run = _run(client, _inventory())
    assert (run.prompt_tokens, run.completion_tokens) == (60, 8)
    assert run.latency_ms >= 0 and run.model == MODEL


def test_to_run_result_records_trace():
    client = FakeAnthropicAgent([[tool_use_block("lookup", {"sku": "FLT-204"}, id="t1")], "Out of stock."])
    result = _run(client, _inventory()).to_run_result("case-1")
    assert result.case_id == "case-1" and result.output == "Out of stock." and result.error is None
    assert result.trace["tool_calls"] == [
        {"name": "lookup", "args": {"sku": "FLT-204"}, "result": '{"sku": "FLT-204", "stock": 0}', "error": None, "step": 1}
    ]
    assert result.trace["steps"] == 2 and result.trace["stop_reason"] == "final_answer"
    assert result.trace["provider"] == "anthropic" and result.trace["model"] == MODEL
    assert (result.prompt_tokens, result.completion_tokens) == (40, 16)


# -- multi-step ------------------------------------------------------------------------------


def test_multi_step_sequential_calls():
    client = FakeAnthropicAgent(
        [
            [tool_use_block("lookup", {"sku": "BRK-001"}, id="a")],
            [tool_use_block("lookup", {"sku": "FLT-204"}, id="b")],
            [tool_use_block("add", {"a": 12, "b": 0}, id="c")],
            "12 parts in total.",
        ]
    )
    tools = _inventory()
    run = _run(client, tools)

    assert run.stop_reason == "final_answer" and run.output == "12 parts in total."
    assert run.steps == 4
    assert [(c.name, c.step) for c in run.tool_calls] == [("lookup", 1), ("lookup", 2), ("add", 3)]
    assert run.tool_calls[2].result == "12"
    # Each request carries the whole conversation so far.
    assert [len(c["messages"]) for c in client.calls] == [1, 3, 5, 7]
    assert _tool_results(client.calls[3])[0]["tool_use_id"] == "c"


def test_parallel_tool_calls_in_one_turn():
    client = FakeAnthropicAgent(
        [
            [tool_use_block("lookup", {"sku": "BRK-001"}, id="x"), tool_use_block("lookup", {"sku": "FLT-204"}, id="y")],
            "BRK-001: 12, FLT-204: 0.",
        ]
    )
    tools = _inventory()
    run = _run(client, tools)

    assert run.steps == 2
    assert [c.step for c in run.tool_calls] == [1, 1]
    assert tools.calls == [("lookup", {"sku": "BRK-001"}), ("lookup", {"sku": "FLT-204"})]
    assert [r["tool_use_id"] for r in _tool_results(client.calls[1])] == ["x", "y"]


# -- tool errors -----------------------------------------------------------------------------


def test_tool_reported_error_is_recorded_and_model_recovers():
    client = FakeAnthropicAgent(
        [
            [tool_use_block("lookup", {"sku": "NOPE-9"}, id="bad")],
            [tool_use_block("lookup", {"sku": "BRK-001"}, id="good")],
            "12 in stock.",
        ]
    )
    run = _run(client, _inventory())

    assert run.stop_reason == "final_answer" and run.error is None and run.output == "12 in stock."
    bad, good = run.tool_calls
    assert bad.error == "unknown sku: NOPE-9" and bad.result == "unknown sku: NOPE-9"
    assert good.error is None
    assert _tool_results(client.calls[1]) == [
        {"type": "tool_result", "tool_use_id": "bad", "content": "unknown sku: NOPE-9", "is_error": True}
    ]
    assert "is_error" not in _tool_results(client.calls[2])[0]


def test_tool_exception_is_reported_back_not_raised():
    def boom(**_kw):
        raise TimeoutError("tool call timed out after 5s")

    client = FakeAnthropicAgent([[tool_use_block("slow", {}, id="s")], "Sorry, the tool timed out."])
    run = _run(client, FakeToolExecutor({"slow": boom}))

    assert run.stop_reason == "final_answer" and run.output == "Sorry, the tool timed out."
    [rec] = run.tool_calls
    assert rec.error == "TimeoutError: tool call timed out after 5s" and rec.result == ""
    [res] = _tool_results(client.calls[1])
    assert res["is_error"] is True and res["content"] == "Error: TimeoutError: tool call timed out after 5s"


def test_unknown_tool_is_an_error():
    client = FakeAnthropicAgent([[tool_use_block("delete_everything", {})], "I can't do that."])
    run = _run(client, _inventory())
    assert run.tool_calls[0].error == "Unknown tool: delete_everything"
    assert _tool_results(client.calls[1])[0]["is_error"] is True


@pytest.mark.parametrize(
    ("raw", "message"),
    [("{not json", "invalid JSON arguments"), ([1, 2], "arguments must be an object, got list")],
)
def test_invalid_arguments_are_not_executed(raw, message):
    client = FakeAnthropicAgent([[tool_use_block("lookup", raw)], "Giving up."])
    tools = _inventory()
    run = _run(client, tools)
    assert tools.calls == []
    assert run.tool_calls[0].args == {} and message in run.tool_calls[0].error
    assert _tool_results(client.calls[1])[0]["is_error"] is True


def test_json_string_arguments_are_parsed():
    client = FakeAnthropicAgent([[tool_use_block("lookup", '{"sku": "BRK-001"}')], "12."])
    tools = _inventory()
    run = _run(client, tools)
    assert tools.calls == [("lookup", {"sku": "BRK-001"})] and run.tool_calls[0].error is None


def test_empty_tool_output_is_sent_as_placeholder():
    client = FakeAnthropicAgent([[tool_use_block("noop", None)], "Done."])
    run = _run(client, FakeToolExecutor({"noop": lambda: ""}))
    assert run.tool_calls[0].args == {}
    assert _tool_results(client.calls[1])[0]["content"] == "(no output)"


# -- max_steps -------------------------------------------------------------------------------


def test_max_steps_hit():
    client = FakeAnthropicAgent(
        [[text_block("Checking again."), tool_use_block("lookup", {"sku": "BRK-001"})]], repeat_last=True
    )
    tools = _inventory()
    run = _run(client, tools, max_steps=3)

    assert run.stop_reason == "max_steps"
    assert run.error == "max_steps (3) reached without a final answer"
    assert run.output == "Checking again."
    assert run.steps == 3 and len(client.calls) == 3 and len(tools.calls) == 3
    assert [c.step for c in run.tool_calls] == [1, 2, 3]
    assert run.to_run_result("c").error == run.error


def test_final_answer_on_last_allowed_step_is_not_max_steps():
    client = FakeAnthropicAgent([[tool_use_block("lookup", {"sku": "BRK-001"})], "12."])
    run = _run(client, _inventory(), max_steps=2)
    assert run.stop_reason == "final_answer" and run.error is None and run.steps == 2


def test_max_steps_must_be_positive():
    with pytest.raises(ValueError, match="max_steps"):
        _run(FakeAnthropicAgent([]), _inventory(), max_steps=0)


# -- provider / setup failures ---------------------------------------------------------------


def test_transient_provider_errors_are_retried():
    client = FakeAnthropicAgent([FakeAPIError(529, retry_after=0), "Fine."])
    run = _run(client, _inventory())
    assert run.stop_reason == "final_answer" and run.output == "Fine."
    assert run.retries == 1 and run.steps == 1 and len(client.calls) == 2


def test_provider_error_ends_run_with_last_text():
    client = FakeAnthropicAgent(
        [[text_block("Looking up."), tool_use_block("lookup", {"sku": "BRK-001"})], FakeAPIError(400)]
    )
    run = _run(client, _inventory())
    assert run.stop_reason == "error" and run.error == "FakeAPIError: HTTP 400"
    assert run.output == "Looking up." and run.steps == 1 and len(run.tool_calls) == 1


def test_list_tools_failure_ends_run():
    class Broken(FakeToolExecutor):
        def list_tools(self):
            raise RuntimeError("server gone")

    client = FakeAnthropicAgent([])
    run = _run(client, Broken({}))
    assert run.stop_reason == "error" and run.error == "listing tools failed: RuntimeError: server gone"
    assert client.calls == []


# -- end to end with the in-repo fake MCP server ---------------------------------------------


def test_agent_against_fake_mcp_server():
    pytest.importorskip("mcp")
    from ai_eval.mcp_client import MCPClient, MCPServerConfig

    client = FakeAnthropicAgent(
        [
            [tool_use_block("lookup", {"sku": "BRK-001"}, id="l1"), tool_use_block("lookup", {"sku": "XXX"}, id="l2")],
            [tool_use_block("add", {"a": 12, "b": 3}, id="a1")],
            "15.",
        ]
    )
    config = MCPServerConfig(command="python", args=[str(FAKE_SERVER)], startup_timeout=20, call_timeout=5)
    with MCPClient(config) as tools:
        run = _run(client, tools)

    assert run.stop_reason == "final_answer" and run.output == "15."
    assert {t["name"] for t in client.calls[0]["tools"]} >= {"echo", "add", "lookup"}
    ok, missing, added = run.tool_calls
    assert ok.error is None and "brake pad" in ok.result
    assert missing.error and "lookup" in missing.error  # tool-reported error from the server
    assert added.error is None and "15" in added.result
