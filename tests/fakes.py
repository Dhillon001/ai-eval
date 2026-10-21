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
