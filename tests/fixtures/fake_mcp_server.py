"""Tiny stdio MCP server used by the test suite (never touches the network).

Run as ``python tests/fixtures/fake_mcp_server.py``. Tools:

- ``echo(text)``            -> the text back
- ``add(a, b)``             -> structured ``{"sum": a + b}``
- ``lookup(sku)``           -> a part record, or a tool error for unknown SKUs
- ``env_value(name)``       -> value of an environment variable seen by the server
- ``slow(seconds)``         -> sleeps, then returns "done" (for timeout tests)
- ``crash()``               -> exits the server process abruptly

Set ``FAKE_MCP_STARTUP_DELAY`` (seconds) to delay startup and ``FAKE_MCP_FAIL_STARTUP=1`` to
exit with an error before serving.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

PARTS = {
    "BRK-001": {"sku": "BRK-001", "name": "brake pad", "stock": 12},
    "FLT-204": {"sku": "FLT-204", "name": "oil filter", "stock": 0},
}

app = MCPServer("fake-mcp")


class Sum(BaseModel):
    sum: int


@app.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return text


@app.tool()
def add(a: int, b: int) -> Sum:
    """Add two integers."""
    return Sum(sum=a + b)


@app.tool()
def lookup(sku: str) -> dict:
    """Look up a part by SKU."""
    if sku not in PARTS:
        raise ValueError(f"unknown sku: {sku}")
    return PARTS[sku]


@app.tool()
def env_value(name: str) -> str:
    """Return an environment variable as seen by the server."""
    return os.environ.get(name, "<unset>")


@app.tool()
async def slow(seconds: float) -> str:
    """Sleep for the given number of seconds, then answer."""
    await asyncio.sleep(seconds)
    return "done"


@app.tool()
def crash() -> str:
    """Kill the server process without replying."""
    sys.stdout.flush()
    os._exit(3)


if __name__ == "__main__":
    if os.environ.get("FAKE_MCP_FAIL_STARTUP") == "1":
        print("fake server refusing to start", file=sys.stderr)
        sys.exit(2)
    delay = float(os.environ.get("FAKE_MCP_STARTUP_DELAY", "0") or 0)
    if delay:
        time.sleep(delay)
    app.run()
