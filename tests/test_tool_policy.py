"""Agent safety limits: tool allowlist/denylist, per-call timeout and max steps from config."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeAnthropicAgent, FakeToolExecutor, tool_use_block

from ai_eval.adapters import build_adapter
from ai_eval.adapters.mcp import MCPAdapter
from ai_eval.agent_loop import ToolPolicy, run_anthropic_agent
from ai_eval.dataset import load_suite
from ai_eval.models import EvalCase
from ai_eval.scorers.tool_use import ToolUseScorer

ROOT = Path(__file__).resolve().parents[1]
FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"
MODEL = "claude-test"


def _tools() -> FakeToolExecutor:
    return FakeToolExecutor(
        {
            "get_part": lambda sku: f'{{"sku": "{sku}"}}',
            "check_stock": lambda sku: '{"stock": 3}',
            "delete_part": lambda sku: "deleted",
            "delete_all": lambda: "wiped",
        }
    )


def _run(client, tools, **kw):
    kw.setdefault("sleep", lambda _s: None)
    return run_anthropic_agent(client, tools, model=MODEL, user="q", **kw)


def _offered(call: dict) -> set[str]:
    return {t["name"] for t in call.get("tools", [])}


# -- ToolPolicy ---------------------------------------------------------------------------------


def test_default_policy_allows_everything():
    policy = ToolPolicy()
    assert policy.is_allowed("anything") and policy.tool_timeout is None


def test_allowlist_and_denylist_with_globs():
    policy = ToolPolicy(allowed_tools=["get_*", "check_stock"], denied_tools=["get_secret"])
    assert policy.is_allowed("get_part") and policy.is_allowed("check_stock")
    assert "not in the allowed tools" in policy.denial_reason("delete_part")
    assert "denied by the tool policy" in policy.denial_reason("get_secret")  # deny wins over allow
    assert ToolPolicy(denied_tools=["delete_*"]).denial_reason("delete_all")
    assert ToolPolicy(allowed_tools=[]).denial_reason("get_part")  # empty allowlist allows nothing


@pytest.mark.parametrize(
    "kwargs",
    [{"allowed_tools": "get_part"}, {"denied_tools": [""]}, {"denied_tools": [3]}, {"tool_timeout": 0}, {"tool_timeout": -1}],
)
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ToolPolicy(**kwargs)


# -- agent loop ---------------------------------------------------------------------------------


def test_disallowed_tools_are_not_offered_to_the_model():
    client = FakeAnthropicAgent(["done"])
    _run(client, _tools(), policy=ToolPolicy(denied_tools=["delete_*"]))
    assert _offered(client.calls[0]) == {"get_part", "check_stock"}

    client = FakeAnthropicAgent(["done"])
    _run(client, _tools(), policy=ToolPolicy(allowed_tools=["get_part"]))
    assert _offered(client.calls[0]) == {"get_part"}


def test_disallowed_call_is_blocked_recorded_and_reported_back():
    tools = _tools()
    client = FakeAnthropicAgent(
        [
            [tool_use_block("delete_part", {"sku": "BRK-001"}, id="d1"), tool_use_block("get_part", {"sku": "BRK-001"}, id="g1")],
            "Looked it up; deleting is not permitted.",
        ]
    )
    run = _run(client, tools, policy=ToolPolicy(denied_tools=["delete_*"]))

    assert run.stop_reason == "final_answer" and run.error is None
    blocked, ok = run.tool_calls
    assert blocked.blocked and blocked.name == "delete_part" and blocked.args == {"sku": "BRK-001"}
    assert "denied by the tool policy" in blocked.error and blocked.result == ""
    assert not ok.blocked and ok.error is None
    assert tools.calls == [("get_part", {"sku": "BRK-001"})]  # the blocked call never ran

    d1, g1 = client.calls[1]["messages"][-1]["content"]
    assert d1["tool_use_id"] == "d1" and d1["is_error"] is True and "denied" in d1["content"]
    assert g1["tool_use_id"] == "g1" and "is_error" not in g1

    trace = run.to_run_result("c1").trace["tool_calls"]
    assert trace[0]["blocked"] is True and "denied" in trace[0]["error"]
    assert "blocked" not in trace[1]  # only blocked calls carry the flag


def test_call_outside_allowlist_is_blocked():
    tools = _tools()
    client = FakeAnthropicAgent([[tool_use_block("check_stock", {"sku": "X"})], "ok"])
    run = _run(client, tools, policy=ToolPolicy(allowed_tools=["get_part"]))
    assert run.tool_calls[0].blocked and "not in the allowed tools" in run.tool_calls[0].error
    assert tools.calls == []


def test_blocked_calls_are_not_counted_as_invalid_arguments():
    case = EvalCase(id="c", input="q", metadata={"expected_tools": [{"name": "get_part"}]})
    client = FakeAnthropicAgent([[tool_use_block("delete_all", {}), tool_use_block("get_part", {"sku": "A"})], "ok"])
    result = _run(client, _tools(), policy=ToolPolicy(denied_tools=["delete_*"])).to_run_result("c")
    score = ToolUseScorer().score(case, result)
    assert score.details["invalid_arg_rate"] == 0.0
    assert score.details["precision"] == 0.5  # the attempted forbidden call still costs precision


def test_tool_timeout_is_passed_to_the_executor():
    tools = _tools()
    client = FakeAnthropicAgent([[tool_use_block("get_part", {"sku": "A"})], "ok"])
    _run(client, tools, policy=ToolPolicy(tool_timeout=2.5))
    assert tools.timeouts == [2.5]

    tools = _tools()
    _run(FakeAnthropicAgent([[tool_use_block("get_part", {"sku": "A"})], "ok"]), tools)
    assert tools.timeouts == [None]


def test_executor_without_timeout_parameter_still_works():
    class Legacy(FakeToolExecutor):
        def call_tool(self, name, arguments=None):  # no timeout kwarg
            return super().call_tool(name, arguments)

    tools = Legacy({"get_part": lambda sku: "ok"})
    run = _run(FakeAnthropicAgent([[tool_use_block("get_part", {"sku": "A"})], "ok"]), tools, policy=ToolPolicy(tool_timeout=1))
    assert run.tool_calls[0].error is None and tools.calls == [("get_part", {"sku": "A"})]


def test_tool_timeout_against_fake_mcp_server():
    pytest.importorskip("mcp")
    from ai_eval.mcp_client import MCPClient, MCPServerConfig

    client = FakeAnthropicAgent(
        [[tool_use_block("slow", {"seconds": 5}, id="s1")], [tool_use_block("echo", {"text": "hi"}, id="e1")], "gave up"]
    )
    config = MCPServerConfig(command="python", args=[str(FAKE_SERVER)], startup_timeout=20, call_timeout=30)
    with MCPClient(config) as tools:
        run = _run(client, tools, policy=ToolPolicy(tool_timeout=0.5))

    slow, echo = run.tool_calls
    assert slow.error and "tim" in slow.error.lower()  # per-call timeout overrides call_timeout=30
    assert echo.error is None and "hi" in echo.result  # the server is still usable afterwards
    assert run.stop_reason == "final_answer"


# -- adapter config -----------------------------------------------------------------------------


def test_adapter_builds_policy_from_config():
    adapter = build_adapter(
        "mcp",
        {
            "server": {"command": "python"},
            "max_steps": 2,
            "allowed_tools": ["get_*"],
            "denied_tools": ["get_secret"],
            "tool_timeout": 4,
        },
    )
    assert adapter.policy == ToolPolicy(allowed_tools=["get_*"], denied_tools=["get_secret"], tool_timeout=4)
    assert adapter.max_steps == 2
    assert build_adapter("mcp", {"server": {"command": "python"}}).policy == ToolPolicy()


@pytest.mark.parametrize(
    "bad", [{"allowed_tools": "get_part"}, {"denied_tools": [None]}, {"tool_timeout": 0}]
)
def test_adapter_rejects_invalid_policy(bad):
    with pytest.raises(ValueError, match="invalid mcp tool policy"):
        build_adapter("mcp", {"server": {"command": "python"}, **bad})


def test_adapter_applies_policy_and_max_steps():
    tools = _tools()
    client = FakeAnthropicAgent([[tool_use_block("delete_all", {})]], repeat_last=True)
    adapter = MCPAdapter(client=client, tools=tools, denied_tools=["delete_*"], max_steps=2, tool_timeout=3)

    result = adapter.run(EvalCase(id="c", input="wipe it"))

    assert "max_steps (2)" in result.error and result.trace["stop_reason"] == "max_steps"
    assert [c["blocked"] for c in result.trace["tool_calls"]] == [True, True]
    assert tools.calls == []
    assert "delete_all" not in _offered(client.calls[0])


def test_example_suite_policy_covers_the_inventory_tools():
    suite = load_suite(ROOT / "configs" / "mcp_suite.yaml", project_root=ROOT)
    opts = dict(suite.adapter)
    opts.pop("type")
    adapter = build_adapter("mcp", {**opts, "root": str(ROOT)})
    assert adapter.policy.tool_timeout == 10
    for name in ("list_warehouses", "search_parts", "get_part", "check_stock", "reserve_part"):
        assert adapter.policy.is_allowed(name)
