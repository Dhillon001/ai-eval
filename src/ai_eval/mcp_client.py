"""Synchronous client for stdio MCP (Model Context Protocol) servers.

The official ``mcp`` SDK is async; :class:`MCPClient` runs it on a private event loop in a
background thread so the (synchronous) runner and adapters can list and call tools with plain
method calls. Install the optional dependency with ``pip install "ai-eval[mcp]"``.

The server is configured in the suite YAML under ``adapter.server``::

    adapter:
      type: mcp
      server:
        command: python            # "python"/"python3" resolve to the running interpreter
        args: [examples/mcp_inventory_server.py]
        env:
          INVENTORY_TOKEN: "${INVENTORY_TOKEN}"   # ${VAR} is read from the environment
        cwd: .
        startup_timeout: 30
        call_timeout: 30
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_eval.adapters.http import HTTPAdapterError, interpolate_env

_PYTHON_ALIASES = frozenset({"python", "python3"})


class MCPClientError(Exception):
    """Transport or protocol failure talking to an MCP server (not a tool-reported error)."""


class MCPTimeoutError(MCPClientError):
    pass


class MCPServerConfig(BaseModel):
    """How to launch a stdio MCP server."""

    model_config = ConfigDict(extra="forbid")

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    startup_timeout: float = Field(default=30.0, gt=0)
    call_timeout: float = Field(default=30.0, gt=0)

    @classmethod
    def from_adapter_options(cls, options: dict[str, Any]) -> MCPServerConfig:
        """Read the ``server`` block of a suite's ``adapter`` section."""
        server = options.get("server")
        if not isinstance(server, dict):
            raise ValueError("mcp adapter requires a 'server' mapping with at least a 'command'")
        return cls.model_validate(server)

    def resolved_command(self) -> str:
        # Run Python servers with the current interpreter so they see the same environment.
        return sys.executable if self.command in _PYTHON_ALIASES else self.command

    def resolved_env(self, env: dict[str, str] | None = None) -> dict[str, str]:
        """Interpolate ``${VAR}`` in env values; a missing variable raises :class:`MCPClientError`."""
        try:
            return {k: interpolate_env(str(v), env) for k, v in self.env.items()}
        except HTTPAdapterError as e:
            raise MCPClientError(f"server env: {e}") from e


class MCPTool(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPToolResult(BaseModel):
    """Outcome of a tool call. ``is_error`` is set when the *tool* reported a failure."""

    text: str = ""
    content: list[dict[str, Any]] = Field(default_factory=list)
    structured: Any = None
    is_error: bool = False


def _import_sdk() -> Any:
    try:
        import mcp
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'MCP support requires the optional dependency: pip install "ai-eval[mcp]"'
        ) from e
    return mcp, StdioServerParameters, stdio_client


def _unwrap(exc: BaseException) -> BaseException:
    """Reduce anyio/asyncio exception groups with a single leaf to that leaf."""
    # Duck-typed so it also matches the ``exceptiongroup`` backport on Python 3.10.
    while isinstance(getattr(exc, "exceptions", None), tuple) and len(exc.exceptions) == 1:  # type: ignore[attr-defined]
        exc = exc.exceptions[0]  # type: ignore[attr-defined]
    return exc


def _content_text(content: list[dict[str, Any]], structured: Any) -> str:
    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
    if texts:
        return "\n".join(texts)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    return ""


class MCPClient:
    """Connect to one stdio MCP server, list its tools, call them, and shut it down cleanly.

    Use as a context manager (``with MCPClient(cfg) as client: ...``) or call :meth:`start` and
    :meth:`close` explicitly. Methods are thread-safe: calls are serialised onto the client's loop.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._stop: asyncio.Event | None = None
        self._ready: Future[None] = Future()
        self._done: Future[None] = Future()
        self._errlog: Any = None
        self._tools: list[MCPTool] | None = None
        self._lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._session is not None and not self._done.done()

    def start(self) -> MCPClient:
        with self._lock:
            if self._thread is not None:
                if self._done.done():
                    raise MCPClientError("client was closed; create a new MCPClient")
                return self
            env = self.config.resolved_env()
            mcp, params_cls, stdio_client = _import_sdk()
            params = params_cls(
                command=self.config.resolved_command(),
                args=list(self.config.args),
                env=env or None,
                cwd=self.config.cwd,
            )
            # Server stderr goes to a temp file so a failed startup can report it.
            self._errlog = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_loop, args=(params, stdio_client, mcp), name="ai-eval-mcp", daemon=True
            )
            self._thread.start()
        try:
            self._ready.result(timeout=self.config.startup_timeout)
        except FutureTimeoutError:
            hint = self._stderr_hint()
            self.close()
            raise MCPTimeoutError(
                f"MCP server did not initialise within {self.config.startup_timeout}s{hint}"
            ) from None
        except BaseException as e:
            hint = self._stderr_hint()
            self.close()
            if isinstance(e, MCPClientError):
                raise
            root = _unwrap(e)
            raise MCPClientError(
                f"failed to start MCP server {self.config.command!r}: {type(root).__name__}: {root}{hint}"
            ) from e
        return self

    def close(self) -> None:
        """Ask the session to exit, wait for the server process to stop, and stop the loop."""
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        if not loop.is_closed() and self._stop is not None:
            loop.call_soon_threadsafe(self._stop.set)
        thread.join(timeout=self.config.startup_timeout + 5)
        self._session = None
        if self._errlog is not None:
            self._errlog.close()
            self._errlog = None

    def __enter__(self) -> MCPClient:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _run_loop(self, params: Any, stdio_client: Any, mcp: Any) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve(params, stdio_client, mcp))
        except BaseException as e:  # noqa: BLE001 - surfaced to start()
            if not self._ready.done():
                self._ready.set_exception(e)
        finally:
            if not self._ready.done():
                self._ready.set_exception(MCPClientError("MCP server exited during startup"))
            self._session = None
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            finally:
                self._loop.close()
                if not self._done.done():
                    self._done.set_result(None)

    async def _serve(self, params: Any, stdio_client: Any, mcp: Any) -> None:
        self._stop = asyncio.Event()
        async with stdio_client(params, errlog=self._errlog) as (read, write):
            # The session-wide read timeout bounds the initialize handshake (startup_timeout);
            # list/call requests are bounded by call_timeout in _submit and call_tool.
            async with mcp.ClientSession(read, write, read_timeout_seconds=self.config.startup_timeout) as session:
                await session.initialize()
                self._session = session
                self._ready.set_result(None)
                await self._stop.wait()

    def _stderr_hint(self) -> str:
        if self._errlog is None or self._errlog.closed:
            return ""
        try:
            self._errlog.seek(0)
            tail = self._errlog.read()[-500:].strip()
        except (OSError, ValueError):
            return ""
        return f"\nserver stderr:\n{tail}" if tail else ""

    # -- requests --------------------------------------------------------------------------

    def _submit(self, coro: Any, timeout: float, wait: float | None = None) -> Any:
        if not self.started or self._loop is None:
            coro.close()
            raise MCPClientError("MCP client is not running; call start() first")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout if wait is None else wait)
        except FutureTimeoutError:
            fut.cancel()
            raise MCPTimeoutError(f"MCP request timed out after {timeout}s") from None
        except MCPClientError:
            raise
        except Exception as exc:  # noqa: BLE001 - protocol/transport errors from the SDK
            e = _unwrap(exc)
            if "timed out" in str(e).lower():
                raise MCPTimeoutError(f"MCP request timed out after {timeout}s") from exc
            raise MCPClientError(f"{type(e).__name__}: {e}") from exc

    def list_tools(self, refresh: bool = False) -> list[MCPTool]:
        """All tools the server exposes (follows pagination; cached after the first call)."""
        if self._tools is not None and not refresh:
            return list(self._tools)

        async def _list() -> list[MCPTool]:
            from mcp import types

            tools: list[MCPTool] = []
            cursor: str | None = None
            while True:
                params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
                result = await self._session.list_tools(params=params)
                for t in result.tools:
                    tools.append(
                        MCPTool(name=t.name, description=t.description or "", input_schema=dict(t.input_schema or {}))
                    )
                cursor = getattr(result, "next_cursor", None)
                if not cursor:
                    return tools

        self._tools = self._submit(_list(), self.config.call_timeout)
        return list(self._tools)

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, timeout: float | None = None
    ) -> MCPToolResult:
        """Call a tool. Tool-reported failures come back with ``is_error=True``; protocol errors
        (unknown tool, bad request, server crash) raise :class:`MCPClientError` and timeouts raise
        :class:`MCPTimeoutError`."""
        limit = self.config.call_timeout if timeout is None else timeout

        async def _call() -> MCPToolResult:
            result = await self._session.call_tool(name, arguments or {}, read_timeout_seconds=limit)
            content = [c.model_dump(mode="json", exclude_none=True) for c in (getattr(result, "content", None) or [])]
            structured = getattr(result, "structured_content", None)
            return MCPToolResult(
                text=_content_text(content, structured),
                content=content,
                structured=structured,
                is_error=bool(getattr(result, "is_error", False)),
            )

        # Small grace period so the SDK's own read timeout fires first with a clearer message.
        return self._submit(_call(), limit, wait=limit + 1.0)
