"""Fake SDK clients so provider code is tested without network or API keys."""

from types import SimpleNamespace


class FakeOpenAI:
    def __init__(self, text="ok", fail=False):
        self.calls = []
        self._text, self._fail = text, fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self._fail:
            raise RuntimeError("boom")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._text))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class FakeAnthropic:
    def __init__(self, text="ok"):
        self.calls = []
        self._text = text
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
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
