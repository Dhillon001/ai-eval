"""MCPClient against the in-repo fake stdio server (tests/fixtures/fake_mcp_server.py)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from ai_eval.mcp_client import (
    MCPClient,
    MCPClientError,
    MCPServerConfig,
    MCPTimeoutError,
)

SERVER = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


def _config(**kw) -> MCPServerConfig:
    kw.setdefault("startup_timeout", 20)
    kw.setdefault("call_timeout", 5)
    return MCPServerConfig(command="python", args=[str(SERVER)], **kw)


@pytest.fixture(scope="module")
def client(_greeting_env: None) -> Iterator[MCPClient]:
    with MCPClient(_config(env={"FAKE_MCP_GREETING": "${FAKE_MCP_TEST_GREETING}"})) as c:
        yield c


@pytest.fixture(scope="module")
def _greeting_env() -> Iterator[None]:
    mp = pytest.MonkeyPatch()
    mp.setenv("FAKE_MCP_TEST_GREETING", "hello from env")
    yield
    mp.undo()


# -- list ------------------------------------------------------------------------------------


def test_list_tools(client: MCPClient):
    tools = {t.name: t for t in client.list_tools()}
    assert set(tools) == {"echo", "add", "lookup", "env_value", "slow", "crash"}
    assert tools["add"].description == "Add two integers."
    schema = tools["add"].input_schema
    assert schema["type"] == "object"
    assert sorted(schema["required"]) == ["a", "b"]
    assert schema["properties"]["a"]["type"] == "integer"


def test_list_tools_is_cached_and_refreshable(client: MCPClient):
    first = client.list_tools()
    first.clear()  # callers get a copy, not the cache
    assert len(client.list_tools()) == 6
    assert [t.name for t in client.list_tools(refresh=True)] == [t.name for t in client.list_tools()]


# -- call ------------------------------------------------------------------------------------


def test_call_text_tool(client: MCPClient):
    result = client.call_tool("echo", {"text": "brake pads"})
    assert result.text == "brake pads"
    assert result.is_error is False
    assert result.content == [{"type": "text", "text": "brake pads"}]


def test_call_structured_tool(client: MCPClient):
    result = client.call_tool("add", {"a": 2, "b": 40})
    assert result.structured == {"sum": 42}
    assert '"sum": 42' in result.text
    assert result.is_error is False


def test_call_returns_record(client: MCPClient):
    result = client.call_tool("lookup", {"sku": "BRK-001"})
    assert result.is_error is False
    assert '"brake pad"' in result.text


def test_server_sees_interpolated_env(client: MCPClient):
    assert client.call_tool("env_value", {"name": "FAKE_MCP_GREETING"}).text == "hello from env"


# -- errors ----------------------------------------------------------------------------------


def test_tool_raised_error_is_reported_not_raised(client: MCPClient):
    result = client.call_tool("lookup", {"sku": "NOPE-999"})
    assert result.is_error is True
    assert "lookup" in result.text


def test_invalid_arguments_are_a_tool_error(client: MCPClient):
    result = client.call_tool("add", {"a": "not a number"})
    assert result.is_error is True
    assert "validation error" in result.text


def test_unknown_tool_is_a_tool_error(client: MCPClient):
    result = client.call_tool("does_not_exist", {})
    assert result.is_error is True
    assert "does_not_exist" in result.text


def test_failed_startup_reports_server_stderr():
    with pytest.raises(MCPClientError, match="refusing to start"):
        MCPClient(_config(env={"FAKE_MCP_FAIL_STARTUP": "1"})).start()


def test_server_crash_raises_and_client_stays_unusable():
    with MCPClient(_config()) as c:
        assert c.call_tool("echo", {"text": "up"}).text == "up"
        with pytest.raises(MCPClientError):
            c.call_tool("crash", {})
        with pytest.raises(MCPClientError):
            c.call_tool("echo", {"text": "again"})


# -- timeouts --------------------------------------------------------------------------------


def test_per_call_timeout_and_recovery(client: MCPClient):
    start = time.monotonic()
    with pytest.raises(MCPTimeoutError, match="timed out"):
        client.call_tool("slow", {"seconds": 10}, timeout=0.5)
    assert time.monotonic() - start < 3
    # The session survives a timed-out call.
    assert client.call_tool("echo", {"text": "still alive"}).text == "still alive"


def test_default_call_timeout_from_config():
    with MCPClient(_config(call_timeout=0.5)) as c:
        with pytest.raises(MCPTimeoutError):
            c.call_tool("slow", {"seconds": 10})
        assert c.call_tool("slow", {"seconds": 0}).text == "done"


def test_startup_timeout():
    cfg = _config(startup_timeout=0.5, env={"FAKE_MCP_STARTUP_DELAY": "5"})
    start = time.monotonic()
    with pytest.raises(MCPTimeoutError, match="did not initialise"):
        MCPClient(cfg).start()
    assert time.monotonic() - start < 10


# -- shutdown --------------------------------------------------------------------------------


def test_clean_shutdown_is_idempotent():
    c = MCPClient(_config())
    c.start()
    assert c.started
    c.close()
    c.close()
    assert not c.started
    with pytest.raises(MCPClientError, match="not running"):
        c.list_tools(refresh=True)
