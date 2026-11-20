"""Fake SDK clients so provider code is tested without network or API keys."""

from types import SimpleNamespace


class FakeAPIError(Exception):
    """Looks like an SDK HTTP error: ``status_code`` plus a ``response`` with headers.

    ``retry_after`` sets a ``Retry-After`` header (seconds); ``0`` lets retry tests run
    without sleeping through the real backoff.
    """

    def __init__(self, status_code=500, retry_after=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        headers = {} if retry_after is None else {"retry-after": str(retry_after)}
        self.response = SimpleNamespace(status_code=status_code, headers=headers)


def _maybe_raise(errors):
    """Pop and raise the next queued exception, if any (a fake that fails, then succeeds)."""
    if errors:
        raise errors.pop(0)


class FakeOpenAI:
    """``errors``: exceptions raised by the first calls, in order, before answering normally."""

    def __init__(self, text="ok", fail=False, errors=None):
        self.calls = []
        self._text, self._fail = text, fail
        self._errors = list(errors or [])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self._fail:
            raise RuntimeError("boom")
        _maybe_raise(self._errors)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._text))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class FakeAnthropic:
    """``errors``: exceptions raised by the first calls, in order, before answering normally."""

    def __init__(self, text="ok", errors=None):
        self.calls = []
        self._text = text
        self._errors = list(errors or [])
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        _maybe_raise(self._errors)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._text)],
            usage=SimpleNamespace(input_tokens=12, output_tokens=7),
        )


class FakeHTTPService:
    """An ``httpx.MockTransport`` that records requests and answers with a canned response.

    Pass ``raise_exc`` (e.g. ``httpx.ReadTimeout``) to simulate a transport failure instead.
    """

    def __init__(self, json=None, status_code=200, text=None, raise_exc=None):
        import httpx

        self.requests = []
        self._json, self._status, self._text, self._raise = json, status_code, text, raise_exc
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request):
        import httpx

        self.requests.append(request)
        if self._raise is not None:
            raise self._raise("simulated", request=request)
        if self._text is not None:
            return httpx.Response(self._status, text=self._text)
        return httpx.Response(self._status, json=self._json)


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, input=None, id=None):
    return SimpleNamespace(type="tool_use", id=id or f"toolu_{name}", name=name, input=input)


class FakeAnthropicAgent:
    """Scripted Anthropic client for agent-loop tests.

    ``turns`` is a list of responses, one per ``messages.create`` call; each is a list of content
    blocks (``text_block`` / ``tool_use_block``), a plain string (a text-only final answer), or an
    exception to raise. When the script runs out, ``repeat_last=True`` keeps returning the last
    turn (for max_steps tests); otherwise extra calls fail the test loudly.
    """

    def __init__(self, turns, repeat_last=False, input_tokens=20, output_tokens=8):
        self.calls = []
        self._turns = list(turns)
        self._repeat_last = repeat_last
        self._usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        idx = len(self.calls) - 1
        if idx >= len(self._turns):
            if not (self._repeat_last and self._turns):
                raise AssertionError(f"FakeAnthropicAgent: unexpected call #{idx + 1}")
            idx = len(self._turns) - 1
        turn = self._turns[idx]
        if isinstance(turn, BaseException):
            raise turn
        blocks = [text_block(turn)] if isinstance(turn, str) else list(turn)
        return SimpleNamespace(content=blocks, usage=self._usage)


class FakeToolExecutor:
    """In-memory stand-in for ``MCPClient``: ``tools`` maps name -> callable(**args).

    A callable may return a string (success), a ``(text, True)`` tuple (tool-reported error), or
    raise (transport-style failure). ``calls`` records ``(name, args)`` in order and
    ``timeouts`` the ``timeout`` passed with each call (``None`` when not given).
    """

    def __init__(self, tools, descriptions=None, schemas=None):
        self.tools = dict(tools)
        self.descriptions = descriptions or {}
        self.schemas = schemas or {}
        self.calls = []
        self.timeouts = []

    def list_tools(self):
        from ai_eval.mcp_client import MCPTool

        return [
            MCPTool(name=n, description=self.descriptions.get(n, ""), input_schema=self.schemas.get(n, {}))
            for n in self.tools
        ]

    def call_tool(self, name, arguments=None, timeout=None):
        from ai_eval.mcp_client import MCPToolResult

        args = dict(arguments or {})
        self.calls.append((name, args))
        self.timeouts.append(timeout)
        if name not in self.tools:
            return MCPToolResult(text=f"Unknown tool: {name}", is_error=True)
        out = self.tools[name](**args)
        if isinstance(out, tuple):
            text, is_error = out
            return MCPToolResult(text=text, is_error=is_error)
        return MCPToolResult(text=str(out))
