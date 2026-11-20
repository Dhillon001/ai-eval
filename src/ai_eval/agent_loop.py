"""Tool-using agent loop: expose MCP tools to Claude and run until a final answer.

The loop is provider-facing on one side (an Anthropic ``messages.create`` client, real or fake)
and tool-facing on the other (anything with ``list_tools()`` / ``call_tool(name, args)``, such
as :class:`ai_eval.mcp_client.MCPClient`). Each model turn that asks for tools has them executed
and their results sent back; the loop stops when the model answers without asking for a tool, or
after ``max_steps`` model turns.

Every tool call is recorded as ``{"name", "args", "result", "error"}`` so it can be stored in
``RunResult.trace["tool_calls"]`` and scored by the trajectory / tool-use scorers::

    with MCPClient(server_config) as tools:
        agent = run_anthropic_agent(client, tools, model="claude-haiku-5-5", user="How many M3 bolts?")
    result = agent.to_run_result(case.id)

A :class:`ToolPolicy` limits what the agent may do: an allowlist and/or denylist of tool names
(exact names or ``fnmatch`` globs such as ``delete_*``) and a per-call timeout. Disallowed tools
are never offered to the model; if it calls one anyway the call is not executed, is recorded with
``blocked=True`` and an error, and the error is sent back to the model.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable
from fnmatch import fnmatchcase
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from ai_eval.llm import DEFAULT_MAX_RETRIES, DEFAULT_RETRY_BASE_DELAY, call_with_retries
from ai_eval.mcp_client import MCPTool, MCPToolResult
from ai_eval.models import RunResult

DEFAULT_MAX_STEPS = 8
DEFAULT_AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. Use them when they help answer the "
    "question, then reply with a concise final answer."
)
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


class ToolExecutor(Protocol):
    """What the loop needs from a tool provider; :class:`MCPClient` satisfies it."""

    def list_tools(self) -> list[MCPTool]: ...

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult: ...


class ToolPolicy(BaseModel):
    """Safety limits for an agent run.

    - ``allowed_tools``: if set, only tools matching one of these names/globs may be called.
    - ``denied_tools``: tools matching any of these names/globs may never be called (deny wins
      over allow).
    - ``tool_timeout``: per-call timeout in seconds, passed to the executor's ``call_tool``
      (``MCPClient`` enforces it; it overrides the server's ``call_timeout``).
    """

    allowed_tools: list[str] | None = None
    denied_tools: list[str] = Field(default_factory=list)
    tool_timeout: float | None = Field(default=None, gt=0)

    @field_validator("allowed_tools", "denied_tools", mode="before")
    @classmethod
    def _names(cls, v: Any) -> Any:
        if isinstance(v, str):
            raise ValueError("must be a list of tool names, not a string")
        if v is not None and any(not isinstance(n, str) or not n.strip() for n in v):
            raise ValueError("tool names must be non-empty strings")
        return v

    def denial_reason(self, name: str) -> str | None:
        """Why ``name`` may not be called, or ``None`` if it is allowed."""
        if any(fnmatchcase(name, pat) for pat in self.denied_tools):
            return f"tool {name!r} is denied by the tool policy"
        if self.allowed_tools is not None and not any(fnmatchcase(name, pat) for pat in self.allowed_tools):
            return f"tool {name!r} is not in the allowed tools"
        return None

    def is_allowed(self, name: str) -> bool:
        return self.denial_reason(name) is None


class ToolCallRecord(BaseModel):
    """One requested tool call. ``error`` is set for tool-reported failures, transport errors and
    calls blocked by the :class:`ToolPolicy` (which also set ``blocked=True`` and are not run)."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    blocked: bool = False
    step: int = 0
    id: str | None = None

    def to_trace(self) -> dict[str, Any]:
        """Trace form: no provider id, and ``blocked`` only on blocked calls."""
        return self.model_dump(exclude={"id"} if self.blocked else {"id", "blocked"})


class AgentRun(BaseModel):
    output: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    steps: int = 0
    stop_reason: Literal["final_answer", "max_steps", "error"] = "final_answer"
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retries: int = 0
    latency_ms: float = 0.0
    model: str | None = None
    provider: str = "anthropic"

    def to_run_result(self, case_id: str) -> RunResult:
        return RunResult(
            case_id=case_id,
            output=self.output,
            trace={
                "provider": self.provider,
                "model": self.model,
                "tool_calls": [c.to_trace() for c in self.tool_calls],
                "steps": self.steps,
                "stop_reason": self.stop_reason,
                "retries": self.retries,
            },
            latency_ms=self.latency_ms,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            error=self.error,
        )


def anthropic_tool_specs(tools: list[MCPTool]) -> list[dict[str, Any]]:
    """MCP tool definitions in Anthropic ``tools=[...]`` format."""
    specs = []
    for t in tools:
        schema = dict(t.input_schema) if t.input_schema else dict(_EMPTY_SCHEMA)
        schema.setdefault("type", "object")
        spec: dict[str, Any] = {"name": t.name, "input_schema": schema}
        if t.description:
            spec["description"] = t.description
        specs.append(spec)
    return specs


def _get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _block_to_param(block: Any) -> dict[str, Any] | None:
    """Turn a response content block into a request content block (for the assistant turn)."""
    kind = _get(block, "type", "text")
    if kind == "text":
        return {"type": "text", "text": _get(block, "text", "") or ""}
    if kind == "tool_use":
        return {"type": "tool_use", "id": _get(block, "id"), "name": _get(block, "name"), "input": _get(block, "input") or {}}
    # Other block types (thinking, ...) are passed back as the SDK gave them when possible.
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    return dict(block) if isinstance(block, dict) else None


def _coerce_args(raw: Any) -> tuple[dict[str, Any], str | None]:
    if raw is None:
        return {}, None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as e:
            return {}, f"invalid JSON arguments: {e}"
    if not isinstance(raw, dict):
        return {}, f"arguments must be an object, got {type(raw).__name__}"
    return raw, None


def _add(total: int | None, value: Any) -> int | None:
    if isinstance(value, int):
        return (total or 0) + value
    return total


def _accepts_timeout(tools: ToolExecutor) -> bool:
    try:
        params = inspect.signature(tools.call_tool).parameters
    except (TypeError, ValueError):
        return False
    return "timeout" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def execute_tool(
    tools: ToolExecutor, name: str, args: dict[str, Any], timeout: float | None = None
) -> tuple[str, str | None]:
    """Run one tool call; returns ``(result_text, error)``. Never raises for tool/transport errors.

    ``timeout`` is passed to executors whose ``call_tool`` accepts it (e.g. ``MCPClient``)."""
    try:
        if timeout is not None and _accepts_timeout(tools):
            res = tools.call_tool(name, args, timeout=timeout)  # type: ignore[call-arg]
        else:
            res = tools.call_tool(name, args)
    except Exception as e:  # noqa: BLE001 - any tool-side failure is reported back to the model
        return "", f"{type(e).__name__}: {e}"
    if res.is_error:
        return res.text, res.text or "tool reported an error"
    return res.text, None


def run_anthropic_agent(
    client: Any,
    tools: ToolExecutor,
    model: str,
    user: str,
    system: str = DEFAULT_AGENT_SYSTEM_PROMPT,
    max_steps: int = DEFAULT_MAX_STEPS,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    sleep: Callable[[float], None] = time.sleep,
    policy: ToolPolicy | None = None,
) -> AgentRun:
    """Run Claude with the executor's tools until it answers without a tool call or ``max_steps``
    model turns have been used.

    Tool failures (tool-reported ``is_error``, timeouts, protocol errors, bad arguments) are
    recorded on the call and returned to the model as ``is_error`` tool results so it can recover.
    A provider failure (after retries) ends the run with ``stop_reason="error"``; hitting
    ``max_steps`` sets ``stop_reason="max_steps"`` and an ``error`` message, with the last text the
    model produced as ``output``.

    ``policy`` (a :class:`ToolPolicy`) hides disallowed tools from the model, blocks calls to
    them (recorded with ``blocked=True`` and an error) and sets the per-call timeout.
    """
    policy = policy or ToolPolicy()
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    start = time.perf_counter()
    run = AgentRun(model=model)

    def finish(**updates: Any) -> AgentRun:
        for k, v in updates.items():
            setattr(run, k, v)
        run.latency_ms = (time.perf_counter() - start) * 1000
        return run

    try:
        specs = anthropic_tool_specs([t for t in tools.list_tools() if policy.is_allowed(t.name)])
    except Exception as e:  # noqa: BLE001
        return finish(stop_reason="error", error=f"listing tools failed: {type(e).__name__}: {e}")

    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    last_text = ""
    for step in range(1, max_steps + 1):
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if specs:
            kwargs["tools"] = specs
        try:
            resp, retries = call_with_retries(
                lambda: client.messages.create(**kwargs),
                max_retries=max_retries,
                base_delay=retry_base_delay,
                sleep=sleep,
            )
        except Exception as e:  # noqa: BLE001
            return finish(output=last_text, stop_reason="error", error=f"{type(e).__name__}: {e}")
        run.steps = step
        run.retries += retries
        usage = getattr(resp, "usage", None)
        run.prompt_tokens = _add(run.prompt_tokens, getattr(usage, "input_tokens", None))
        run.completion_tokens = _add(run.completion_tokens, getattr(usage, "output_tokens", None))

        blocks = list(getattr(resp, "content", None) or [])
        last_text = "".join(_get(b, "text", "") or "" for b in blocks if _get(b, "type", "text") == "text")
        tool_uses = [b for b in blocks if _get(b, "type") == "tool_use"]
        if not tool_uses:
            return finish(output=last_text, stop_reason="final_answer")

        messages.append({"role": "assistant", "content": [p for b in blocks if (p := _block_to_param(b))]})
        results: list[dict[str, Any]] = []
        for tu in tool_uses:
            name = str(_get(tu, "name", ""))
            args, arg_error = _coerce_args(_get(tu, "input"))
            denied = policy.denial_reason(name)
            if denied is not None:
                text, error = "", denied
            elif arg_error is None:
                text, error = execute_tool(tools, name, args, timeout=policy.tool_timeout)
            else:
                text, error = "", arg_error
            run.tool_calls.append(
                ToolCallRecord(
                    name=name, args=args, result=text, error=error, blocked=denied is not None,
                    step=step, id=_get(tu, "id"),
                )
            )
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": _get(tu, "id"),
                "content": (f"Error: {error}" if error and not text else text) or "(no output)",
            }
            if error:
                result["is_error"] = True
            results.append(result)
        messages.append({"role": "user", "content": results})

    return finish(
        output=last_text,
        stop_reason="max_steps",
        error=f"max_steps ({max_steps}) reached without a final answer",
    )
