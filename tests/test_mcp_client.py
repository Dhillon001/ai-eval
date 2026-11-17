from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from ai_eval.dataset import load_suite
from ai_eval.mcp_client import MCPClient, MCPClientError, MCPServerConfig

pytest.importorskip("mcp")

SERVER = textwrap.dedent(
    '''
    from mcp.server.mcpserver import MCPServer

    app = MCPServer("smoke")

    @app.tool()
    def echo(text: str) -> str:
        """Echo the text back."""
        return text

    app.run()
    '''
)


def test_config_from_suite_yaml(tmp_path: Path):
    (tmp_path / "suite.yaml").write_text(
        textwrap.dedent(
            """
            name: mcp-smoke
            dataset: data.jsonl
            adapter:
              type: mcp
              server:
                command: python
                args: [server.py, --verbose]
                env: {TOKEN: "${MCP_TEST_TOKEN}", MODE: test}
                cwd: srv
                call_timeout: 5
            """
        ),
        encoding="utf-8",
    )
    suite = load_suite(tmp_path / "suite.yaml", project_root=tmp_path)
    cfg = MCPServerConfig.from_adapter_options(suite.adapter)
    assert cfg.args == ["server.py", "--verbose"]
    assert cfg.cwd == "srv"
    assert cfg.call_timeout == 5
    assert cfg.startup_timeout == 30
    assert cfg.resolved_command() == sys.executable
    assert cfg.resolved_env({"MCP_TEST_TOKEN": "s3cret"}) == {"TOKEN": "s3cret", "MODE": "test"}


def test_config_validation():
    with pytest.raises(ValueError, match="server"):
        MCPServerConfig.from_adapter_options({"type": "mcp"})
    with pytest.raises(ValueError):
        MCPServerConfig.from_adapter_options({"server": {"command": "x", "bogus": 1}})
    with pytest.raises(ValueError):
        MCPServerConfig(command="x", call_timeout=0)
    assert MCPServerConfig(command="node").resolved_command() == "node"


def test_missing_env_var_fails_before_spawning():
    cfg = MCPServerConfig(command="python", env={"TOKEN": "${AI_EVAL_UNSET_VAR_XYZ}"})
    with pytest.raises(MCPClientError, match="AI_EVAL_UNSET_VAR_XYZ"):
        MCPClient(cfg).start()


def test_missing_command_raises_client_error():
    with pytest.raises(MCPClientError, match="failed to start"):
        MCPClient(MCPServerConfig(command="ai-eval-no-such-command-xyz", startup_timeout=5)).start()


def test_requests_require_started_client():
    with pytest.raises(MCPClientError, match="not running"):
        MCPClient(MCPServerConfig(command="python")).list_tools()


def test_list_and_call_roundtrip(tmp_path: Path):
    (tmp_path / "server.py").write_text(SERVER, encoding="utf-8")
    cfg = MCPServerConfig(command="python", args=[str(tmp_path / "server.py")], startup_timeout=20)
    client = MCPClient(cfg)
    with client:
        tools = client.list_tools()
        assert [t.name for t in tools] == ["echo"]
        assert tools[0].description == "Echo the text back."
        assert tools[0].input_schema["required"] == ["text"]
        result = client.call_tool("echo", {"text": "hi"})
        assert result.text == "hi"
        assert result.is_error is False
    assert not client.started
    with pytest.raises(MCPClientError, match="closed"):
        client.start()
