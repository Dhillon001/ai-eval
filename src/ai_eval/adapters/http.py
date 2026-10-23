from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx

from ai_eval.adapters.base import SystemAdapter
from ai_eval.models import EvalCase, RunResult

DEFAULT_BODY_TEMPLATE: dict[str, Any] = {"input": "{input}", "context": "{context}"}

_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_MISSING = object()


class HTTPAdapterError(Exception):
    pass


def interpolate_env(value: str, env: dict[str, str] | None = None) -> str:
    """Replace ``${NAME}`` with the environment variable ``NAME``; a missing variable is an error."""
    source = os.environ if env is None else env

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in source:
            raise HTTPAdapterError(f"environment variable {name} is not set")
        return source[name]

    return _ENV_VAR.sub(_sub, value)


def render_template(template: Any, case: EvalCase) -> Any:
    """Fill ``{input}``/``{context}`` placeholders in a JSON-like template.

    A string that is exactly a placeholder is replaced by the raw value (so ``"{context}"`` becomes
    ``null`` when the case has no context); placeholders embedded in longer strings are substituted
    as text (missing context becomes an empty string). Dicts and lists are rendered recursively.
    """
    values = {"{input}": case.input, "{context}": case.context}
    if isinstance(template, str):
        if template in values:
            return values[template]
        out = template
        for key, val in values.items():
            out = out.replace(key, val if val is not None else "")
        return out
    if isinstance(template, dict):
        return {k: render_template(v, case) for k, v in template.items()}
    if isinstance(template, list):
        return [render_template(v, case) for v in template]
    return template


def extract_path(payload: Any, path: str) -> Any:
    """Follow a dotted path (``data.answer``, ``choices.0.message.content``) into decoded JSON.

    Returns ``_MISSING`` when any segment does not resolve. An empty path returns the payload itself.
    """
    if not path:
        return payload
    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return _MISSING
        else:
            return _MISSING
    return current


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


class HTTPAdapter(SystemAdapter):
    """Evaluate any service that answers a JSON POST.

    The request body is built from ``body_template`` with ``{input}``/``{context}`` placeholders,
    header values may reference environment variables as ``${API_TOKEN}``, and the system output
    is read from the JSON response at the dotted ``response_path``.
    """

    name = "http"

    def __init__(
        self,
        url: str,
        body_template: dict[str, Any] | list[Any] | str | None = None,
        response_path: str = "output",
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        method: str = "POST",
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        if not url:
            raise ValueError("http adapter requires a 'url'")
        self.url = url
        self.body_template = DEFAULT_BODY_TEMPLATE if body_template is None else body_template
        self.response_path = response_path
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.method = method.upper()
        self._client = client
        self._transport = transport

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout, transport=self._transport)
        return self._client

    def _resolved_headers(self) -> dict[str, str]:
        return {k: interpolate_env(str(v)) for k, v in self.headers.items()}

    def build_request_body(self, case: EvalCase) -> Any:
        return render_template(self.body_template, case)

    def run(self, case: EvalCase) -> RunResult:
        trace: dict[str, Any] = {"adapter": "http", "url": self.url, "method": self.method}
        try:
            headers = self._resolved_headers()
        except HTTPAdapterError as e:
            return RunResult(case_id=case.id, output="", trace=trace, error=str(e))

        body = self.build_request_body(case)
        trace["request"] = body
        start = time.perf_counter()

        def _fail(msg: str) -> RunResult:
            return RunResult(
                case_id=case.id,
                output="",
                trace=trace,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=msg,
            )

        try:
            resp = self._get_client().request(
                self.method, self.url, json=body, headers=headers, timeout=self.timeout
            )
        except httpx.TimeoutException as e:
            return _fail(f"timeout after {self.timeout}s: {type(e).__name__}")
        except httpx.HTTPError as e:
            return _fail(f"{type(e).__name__}: {e}")

        trace["status_code"] = resp.status_code
        if not resp.is_success:
            snippet = resp.text[:200]
            return _fail(f"HTTP {resp.status_code}: {snippet}")

        try:
            payload = resp.json()
        except ValueError:
            return _fail(f"response is not valid JSON: {resp.text[:200]}")

        value = extract_path(payload, self.response_path)
        if value is _MISSING:
            return _fail(f"response_path '{self.response_path}' not found in response")

        return RunResult(
            case_id=case.id,
            output=_as_text(value),
            trace=trace,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
