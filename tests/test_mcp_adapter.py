"""The ``mcp`` adapter: config wiring, lifecycle, and end-to-end runs against the example
inventory server (a local stdio subprocess) driven by a scripted fake Anthropic client."""

from __future__ import annotations

import importlib.util
import json
import textwrap
from pathlib import Path

import pytest
from fakes import FakeAnthropicAgent, FakeToolExecutor, text_block, tool_use_block

import ai_eval.adapters.mcp as mcp_adapter_mod
from ai_eval.adapters import ADAPTERS, build_adapter
from ai_eval.adapters.mcp import MCPAdapter
from ai_eval.dataset import load_cases, load_suite
from ai_eval.mcp_client import MCPClient, MCPServerConfig
from ai_eval.models import EvalCase
from ai_eval.runner import run_suite
from ai_eval.store import RunStore

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs" / "mcp_suite.yaml"
INVENTORY_SERVER = {"command": "python", "args": ["examples/mcp_inventory_server.py"], "startup_timeout": 30}


def _case(text: str = "q", cid: str = "c1") -> EvalCase:
    return EvalCase(id=cid, input=text)


# -- wiring -------------------------------------------------------------------------------------


def test_mcp_is_registered_and_validates_options():
    assert "mcp" in ADAPTERS
    adapter = build_adapter("mcp", {"server": {"command": "python"}, "model": "claude-x", "max_steps": 3})
    assert isinstance(adapter, MCPAdapter)
    assert adapter.model == "claude-x"
    assert adapter.max_steps == 3
    with pytest.raises(ValueError, match="Unknown option.*bogus"):
        build_adapter("mcp", {"server": {"command": "python"}, "bogus": 1})
    with pytest.raises(ValueError, match="server"):
        build_adapter("mcp", {})
    with pytest.raises(ValueError, match="provider"):
        build_adapter("mcp", {"server": {"command": "python"}, "provider": "gemini"})
    with pytest.raises(ValueError, match="max_steps"):
        build_adapter("mcp", {"server": {"command": "python"}, "max_steps": 0})


def test_relative_cwd_resolves_against_root(tmp_path: Path):
    adapter = MCPAdapter(server={"command": "python", "cwd": "srv"}, root=tmp_path)
    assert adapter.server_config.cwd == str((tmp_path / "srv").resolve())
    adapter = MCPAdapter(server={"command": "python"}, root=tmp_path)
    assert adapter.server_config.cwd == str(tmp_path.resolve())
    adapter = MCPAdapter(server={"command": "python", "cwd": "srv"})
    assert adapter.server_config.cwd == "srv"


def test_suite_config_builds_adapter():
    suite = load_suite(SUITE, project_root=ROOT)
    opts = dict(suite.adapter)
    assert opts.pop("type") == "mcp"
    adapter = build_adapter("mcp", {**opts, "root": str(ROOT)})
    assert adapter.server_config.cwd == str(ROOT)
    assert adapter.server_config.args == ["examples/mcp_inventory_server.py"]


# -- run with injected fakes --------------------------------------------------------------------


def test_run_records_tool_calls_with_injected_executor():
    tools = FakeToolExecutor({"get_part": lambda sku: json.dumps({"sku": sku, "price_usd": 9.75})})
    client = FakeAnthropicAgent([[tool_use_block("get_part", {"sku": "FLT-204"})], "It costs $9.75."])
    adapter = MCPAdapter(client=client, tools=tools, model="claude-test", system_prompt="be brief")

    result = adapter.run(EvalCase(id="price", input="Price of FLT-204?", context="Prices are USD."))

    assert result.error is None
    assert result.output == "It costs $9.75."
    assert result.trace["adapter"] == "mcp"
    assert result.trace["model"] == "claude-test"
    assert result.trace["stop_reason"] == "final_answer"
    assert result.trace["tool_calls"] == [
        {"name": "get_part", "args": {"sku": "FLT-204"}, "result": '{"sku": "FLT-204", "price_usd": 9.75}', "error": None, "step": 1}
    ]
    assert result.prompt_tokens == 40 and result.completion_tokens == 16
    first = client.calls[0]
    assert first["system"] == "be brief"
    assert "Prices are USD." in first["messages"][0]["content"]  # context goes into the prompt
    adapter.close()  # injected executors are not ours to close
    assert tools.calls == [("get_part", {"sku": "FLT-204"})]


def test_missing_api_key_is_a_case_error(monkeypatch):
    def boom(*_a, **_k):
        raise mcp_adapter_mod.LLMError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(mcp_adapter_mod, "make_client", boom)
    result = MCPAdapter(server={"command": "python"}).run(_case())
    assert result.output == ""
    assert "ANTHROPIC_API_KEY" in result.error
    assert result.trace["adapter"] == "mcp"


def test_failed_server_start_is_remembered(monkeypatch):
    started = []

    class CountingClient(MCPClient):
        def start(self):
            started.append(self.config.command)
            return super().start()

    monkeypatch.setattr(mcp_adapter_mod, "MCPClient", CountingClient)
    adapter = MCPAdapter(
        server={"command": "ai-eval-no-such-command-xyz", "startup_timeout": 5},
        client=FakeAnthropicAgent([]),
    )
    first, second = adapter.run(_case(cid="a")), adapter.run(_case(cid="b"))
    assert "failed to start" in first.error
    assert second.error == first.error
    assert started == ["ai-eval-no-such-command-xyz"]  # no second spawn attempt
    assert first.trace["server"] == "ai-eval-no-such-command-xyz"
    adapter.close()


# -- end to end against the example inventory server --------------------------------------------

needs_mcp = pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="needs the mcp extra")


@needs_mcp
def test_inventory_server_tools_match_dataset():
    cfg = MCPServerConfig(command="python", args=[str(ROOT / "examples/mcp_inventory_server.py")], startup_timeout=30)
    with MCPClient(cfg) as client:
        names = {t.name for t in client.list_tools()}
        stock = client.call_tool("check_stock", {"sku": "BRK-014", "warehouse": "BOS"})
        short = client.call_tool("reserve_part", {"sku": "BRK-014", "quantity": 5, "warehouse": "BOS"})
        missing = client.call_tool("get_part", {"sku": "XYZ-999"})
    assert names == {"list_warehouses", "search_parts", "get_part", "check_stock", "reserve_part"}
    assert json.loads(stock.text) == {"sku": "BRK-014", "warehouse": "BOS", "on_hand": 2}
    assert short.is_error and "insufficient stock" in short.text
    assert missing.is_error and "unknown sku" in missing.text

    cases = load_cases(ROOT / "datasets/examples/mcp_inventory_v1.jsonl")
    assert len(cases) == 8
    assert len({c.id for c in cases}) == len(cases)
    for case in cases:
        assert "mcp" in case.metadata["tags"], case.id
        assert "expected_tools" in case.metadata, case.id
        for call in case.metadata["expected_tools"] + case.metadata["mock_tool_calls"]:
            assert call["name"] in names, (case.id, call["name"])


@needs_mcp
def test_adapter_runs_agent_against_inventory_server():
    client = FakeAnthropicAgent(
        [
            [text_block("Searching."), tool_use_block("search_parts", {"query": "rotor"}, id="t1")],
            [tool_use_block("check_stock", {"sku": "BRK-014", "warehouse": "SEA"}, id="t2")],
            "No, Seattle has 0 vented brake rotors.",
        ]
    )
    adapter = MCPAdapter(server=INVENTORY_SERVER, root=ROOT, client=client)
    try:
        result = adapter.run(_case("Any vented brake rotors in Seattle?"))
    finally:
        adapter.close()

    assert result.error is None, result.error
    assert result.output == "No, Seattle has 0 vented brake rotors."
    calls = result.trace["tool_calls"]
    assert [c["name"] for c in calls] == ["search_parts", "check_stock"]
    assert "BRK-014" in calls[0]["result"]
    assert json.loads(calls[1]["result"]) == {"sku": "BRK-014", "warehouse": "SEA", "on_hand": 0}
    assert result.trace["server"] == "python examples/mcp_inventory_server.py"
    # Tool specs sent to the model come from the server's tool listing.
    assert {t["name"] for t in client.calls[0]["tools"]} >= {"search_parts", "check_stock"}
    assert adapter._owned is None


@needs_mcp
def test_run_suite_with_mcp_adapter(tmp_path: Path, monkeypatch):
    (tmp_path / "data.jsonl").write_text(
        "\n".join(
            json.dumps(c)
            for c in [
                {"id": "price", "input": "Price of BRK-001?", "metadata": {"expected_tools": [{"name": "get_part", "args": {"sku": "BRK-001"}}]}},
                {"id": "unknown", "input": "Price of XYZ-999?", "metadata": {"expected_tools": [{"name": "get_part", "args": {"sku": "XYZ-999"}}]}},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "suite.yaml").write_text(
        textwrap.dedent(
            f"""
            name: mcp-mini
            dataset: data.jsonl
            adapter:
              type: mcp
              model: claude-haiku-5-5
              server:
                command: python
                args: [examples/mcp_inventory_server.py]
                cwd: {ROOT.as_posix()}
            scorers:
              - type: trajectory
                mode: subset
                match_args: true
            """
        ),
        encoding="utf-8",
    )
    fake = FakeAnthropicAgent(
        [
            [tool_use_block("get_part", {"sku": "BRK-001"})],
            "$48.50",
            [tool_use_block("get_part", {"sku": "XYZ-999"})],
            "There is no part XYZ-999.",
        ]
    )
    monkeypatch.setattr(mcp_adapter_mod, "make_client", lambda *_a, **_k: fake)

    _, summary, outcomes = run_suite(
        suite_path=tmp_path / "suite.yaml", project_root=tmp_path, store=RunStore(tmp_path / "runs.db")
    )
    assert summary.adapter == "mcp"
    assert summary.model == "claude-haiku-5-5"
    assert summary.pass_rate == 1.0
    by_id = {o.case_id: o.run for o in outcomes}
    assert "48.5" in by_id["price"].trace["tool_calls"][0]["result"]
    assert by_id["unknown"].trace["tool_calls"][0]["error"]
    assert by_id["unknown"].output == "There is no part XYZ-999."
    # The project root is used to build the adapter but never leaks into the stored options.
    assert "root" not in summary.metadata.get("adapter_options", {})


def test_mcp_suite_runs_offline_with_mock_adapter(tmp_path: Path):
    _, summary, _ = run_suite(
        suite_path=SUITE, project_root=ROOT, adapter_name="mock", store=RunStore(tmp_path / "runs.db")
    )
    assert summary.total == 8
    assert summary.pass_rate == 1.0
