"""Evaluate an MCP server end to end: Claude answers each case using the server's tools.

The adapter launches the stdio server once per run (lazily, on the first case), shares it across
cases (the client serialises calls, so ``--concurrency`` is safe), runs the agent loop for each
case, and shuts the server down when the runner calls :meth:`MCPAdapter.close`. Every tool call is
stored in ``RunResult.trace["tool_calls"]`` for the trajectory / tool-use scorers::

    adapter:
      type: mcp
      model: claude-haiku-5-5
      max_steps: 6
      allowed_tools: [find_part, check_stock, reserve_part]   # optional allowlist (globs ok)
      denied_tools: ["delete_*"]                              # optional denylist (wins over allow)
      tool_timeout: 10                                        # optional per-call timeout (s)
      server:
        command: python
        args: [examples/mcp_inventory_server.py]

Disallowed tools are hidden from the model; calls to them are not executed and are recorded in
the trace with ``blocked: true`` and an error.

Relative ``server.cwd`` (or no ``cwd`` at all) is resolved against the project root, so suites run
the same from any working directory.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ai_eval.adapters.base import SystemAdapter
from ai_eval.agent_loop import (
    DEFAULT_AGENT_SYSTEM_PROMPT,
    DEFAULT_MAX_STEPS,
    ToolExecutor,
    ToolPolicy,
    run_anthropic_agent,
)
from ai_eval.llm import DEFAULT_MAX_RETRIES, LLMError, build_user_prompt, make_client
from ai_eval.mcp_client import MCPClient, MCPServerConfig
from ai_eval.models import EvalCase, RunResult

MCP_OPTIONS = (
    "server", "provider", "model", "base_url", "system_prompt", "max_steps",
    "temperature", "max_tokens", "max_retries", "root",
    "allowed_tools", "denied_tools", "tool_timeout",
)
PROVIDERS = ("anthropic",)


class MCPAdapter(SystemAdapter):
    """Run Claude against a stdio MCP server's tools for each case.

    ``client`` (an Anthropic-style client) and ``tools`` (anything satisfying
    :class:`~ai_eval.agent_loop.ToolExecutor`) can be injected for tests; otherwise they are built
    from the environment (``ANTHROPIC_API_KEY``) and the ``server`` config.
    """

    name = "mcp"
    default_model = "claude-haiku-5-5"

    def __init__(
        self,
        server: dict[str, Any] | MCPServerConfig | None = None,
        provider: str = "anthropic",
        model: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = DEFAULT_MAX_RETRIES,
        root: str | Path | None = None,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
        tool_timeout: float | None = None,
        client: Any = None,
        tools: ToolExecutor | None = None,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"mcp adapter provider must be one of {', '.join(PROVIDERS)}, got {provider!r}")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        try:
            self.policy = ToolPolicy(
                allowed_tools=allowed_tools, denied_tools=denied_tools or [], tool_timeout=tool_timeout
            )
        except ValueError as e:
            raise ValueError(f"invalid mcp tool policy: {e}") from None
        if tools is None:
            if server is None:
                raise ValueError("mcp adapter requires a 'server' mapping with at least a 'command'")
            config = server if isinstance(server, MCPServerConfig) else MCPServerConfig.from_adapter_options({"server": server})
            self.server_config: MCPServerConfig | None = _resolve_cwd(config, root)
        else:
            self.server_config = None
        self.provider = provider
        self.model = model or self.default_model
        self.base_url = base_url
        self.system_prompt = system_prompt or DEFAULT_AGENT_SYSTEM_PROMPT
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = client
        self._tools: Any = tools
        self._owned: MCPClient | None = None
        self._start_error: str | None = None
        self._lock = threading.Lock()

    def _get_tools(self) -> Any:
        """Start the server on first use; a failed start is remembered so later cases fail fast."""
        with self._lock:
            if self._tools is not None:
                return self._tools
            if self._start_error is not None:
                raise RuntimeError(self._start_error)
            assert self.server_config is not None
            try:
                self._owned = MCPClient(self.server_config).start()
            except Exception as e:  # noqa: BLE001 - reported on every case's RunResult
                self._start_error = f"MCP server failed to start: {type(e).__name__}: {e}"
                raise RuntimeError(self._start_error) from e
            self._tools = self._owned
            return self._tools

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = make_client(self.provider, self.base_url)
        return self._client

    def _trace_base(self) -> dict[str, Any]:
        trace: dict[str, Any] = {"adapter": "mcp"}
        if self.server_config is not None:
            trace["server"] = " ".join([self.server_config.command, *self.server_config.args])
        return trace

    def run(self, case: EvalCase) -> RunResult:
        try:
            client = self._get_client()
        except LLMError as e:
            return RunResult(case_id=case.id, output="", trace=self._trace_base(), error=str(e))
        try:
            tools = self._get_tools()
        except RuntimeError as e:
            return RunResult(case_id=case.id, output="", trace=self._trace_base(), error=str(e))

        agent = run_anthropic_agent(
            client,
            tools,
            model=self.model,
            user=build_user_prompt(case.input, case.context),
            system=self.system_prompt,
            max_steps=self.max_steps,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            policy=self.policy,
        )
        result = agent.to_run_result(case.id)
        result.trace = {**self._trace_base(), **result.trace}
        return result

    def close(self) -> None:
        """Shut down the server this adapter started (injected executors are left alone)."""
        with self._lock:
            owned, self._owned = self._owned, None
            if owned is not None:
                self._tools = None
        if owned is not None:
            owned.close()


def _resolve_cwd(config: MCPServerConfig, root: str | Path | None) -> MCPServerConfig:
    if root is None:
        return config
    base = Path(root)
    cwd = Path(config.cwd) if config.cwd else base
    if not cwd.is_absolute():
        cwd = base / cwd
    return config.model_copy(update={"cwd": str(cwd.resolve())})
