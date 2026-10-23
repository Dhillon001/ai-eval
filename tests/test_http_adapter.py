from __future__ import annotations

import json

import httpx
import pytest

from ai_eval.adapters.http import (
    _MISSING,
    HTTPAdapter,
    extract_path,
    interpolate_env,
    render_template,
)
from ai_eval.models import EvalCase
from fakes import FakeHTTPService

URL = "https://svc.example/answer"


def _case(**kw) -> EvalCase:
    return EvalCase(id=kw.pop("id", "c1"), input=kw.pop("input", "refund?"), **kw)


def test_success_posts_rendered_body_and_extracts_dotted_path():
    svc = FakeHTTPService(json={"data": {"answer": "30 days"}})
    adapter = HTTPAdapter(
        URL,
        body_template={"q": "{input}", "ctx": "{context}", "meta": ["Q: {input}"]},
        response_path="data.answer",
        transport=svc.transport,
    )
    r = adapter.run(_case(context="Policy: 30 days"))
    assert r.error is None and r.output == "30 days"
    assert r.latency_ms is not None and r.latency_ms >= 0
    assert r.trace["status_code"] == 200
    req = svc.requests[0]
    assert req.method == "POST" and str(req.url) == URL
    assert json.loads(req.content) == {
        "q": "refund?",
        "ctx": "Policy: 30 days",
        "meta": ["Q: refund?"],
    }


def test_default_template_and_list_index_path():
    svc = FakeHTTPService(json={"choices": [{"message": {"content": "hi"}}]})
    r = HTTPAdapter(URL, response_path="choices.0.message.content", transport=svc.transport).run(
        _case(input="hello")
    )
    assert r.output == "hi"
    assert json.loads(svc.requests[0].content) == {"input": "hello", "context": None}


def test_non_string_output_is_json_encoded():
    svc = FakeHTTPService(json={"output": {"label": "yes", "score": 1}})
    r = HTTPAdapter(URL, transport=svc.transport).run(_case())
    assert json.loads(r.output) == {"label": "yes", "score": 1}


@pytest.mark.parametrize("status", [400, 404, 500, 503])
def test_non_2xx_becomes_error(status):
    svc = FakeHTTPService(text="upstream exploded", status_code=status)
    r = HTTPAdapter(URL, transport=svc.transport).run(_case())
    assert r.output == ""
    assert r.error == f"HTTP {status}: upstream exploded"
    assert r.trace["status_code"] == status


def test_timeout_becomes_error():
    svc = FakeHTTPService(raise_exc=httpx.ReadTimeout)
    r = HTTPAdapter(URL, timeout=2.5, transport=svc.transport).run(_case())
    assert r.output == "" and r.error.startswith("timeout after 2.5s")
    assert "status_code" not in r.trace


def test_connection_error_becomes_error():
    svc = FakeHTTPService(raise_exc=httpx.ConnectError)
    r = HTTPAdapter(URL, transport=svc.transport).run(_case())
    assert r.output == "" and "ConnectError" in r.error


def test_bad_response_path_is_error():
    svc = FakeHTTPService(json={"data": {"text": "x"}})
    r = HTTPAdapter(URL, response_path="data.answer", transport=svc.transport).run(_case())
    assert r.output == "" and "response_path 'data.answer' not found" in r.error


def test_non_json_response_is_error():
    svc = FakeHTTPService(text="<html>oops</html>")
    r = HTTPAdapter(URL, transport=svc.transport).run(_case())
    assert r.output == "" and "not valid JSON" in r.error


def test_header_env_interpolation(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    svc = FakeHTTPService(json={"output": "ok"})
    adapter = HTTPAdapter(
        URL,
        headers={"Authorization": "Bearer ${API_TOKEN}", "X-Static": "plain"},
        transport=svc.transport,
    )
    assert adapter.run(_case()).output == "ok"
    req = svc.requests[0]
    assert req.headers["Authorization"] == "Bearer s3cret"
    assert req.headers["X-Static"] == "plain"


def test_missing_header_env_var_is_error_without_request(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    svc = FakeHTTPService(json={"output": "ok"})
    r = HTTPAdapter(URL, headers={"Authorization": "Bearer ${API_TOKEN}"}, transport=svc.transport).run(
        _case()
    )
    assert r.output == "" and "API_TOKEN is not set" in r.error
    assert svc.requests == []


def test_method_is_configurable():
    svc = FakeHTTPService(json={"output": "ok"})
    HTTPAdapter(URL, method="put", transport=svc.transport).run(_case())
    assert svc.requests[0].method == "PUT"


def test_missing_url_rejected():
    with pytest.raises(ValueError, match="url"):
        HTTPAdapter("")


def test_helpers():
    assert interpolate_env("a-${X}-b", {"X": "1"}) == "a-1-b"
    case = _case(input="q")
    assert render_template("ctx={context}", case) == "ctx="
    assert render_template({"n": 3}, case) == {"n": 3}
    assert extract_path({"a": [1, 2]}, "a.5") is _MISSING
    assert extract_path({"a": "s"}, "a.b") is _MISSING
    assert extract_path({"a": [1, 2]}, "a.1") == 2
    assert extract_path({"a": 1}, "") == {"a": 1}


def test_trace_path_mapping_captures_fields_and_reports_missing():
    svc = FakeHTTPService(
        json={"data": {"answer": "ok", "retrieved_ids": ["d1", "d2"], "tool_calls": [{"name": "search"}]}}
    )
    r = HTTPAdapter(
        URL,
        response_path="data.answer",
        trace_path={
            "retrieved_ids": "data.retrieved_ids",
            "tool_calls": "data.tool_calls",
            "scores": "data.scores",
        },
        transport=svc.transport,
    ).run(_case())
    assert r.error is None and r.output == "ok"
    assert r.trace["retrieved_ids"] == ["d1", "d2"]
    assert r.trace["tool_calls"] == [{"name": "search"}]
    assert r.trace["trace_missing"] == ["data.scores"]


def test_trace_path_string_merges_object_without_clobbering_reserved_keys():
    svc = FakeHTTPService(
        json={"answer": "ok", "debug": {"retrieved_ids": ["d9"], "status_code": 999, "adapter": "x"}}
    )
    r = HTTPAdapter(URL, response_path="answer", trace_path="debug", transport=svc.transport).run(_case())
    assert r.trace["retrieved_ids"] == ["d9"]
    assert r.trace["status_code"] == 200 and r.trace["adapter"] == "http"
    assert "trace_missing" not in r.trace


def test_trace_path_string_not_an_object_is_reported_not_fatal():
    svc = FakeHTTPService(json={"answer": "ok", "debug": ["not", "a", "dict"]})
    r = HTTPAdapter(URL, response_path="answer", trace_path="debug", transport=svc.transport).run(_case())
    assert r.error is None and r.output == "ok"
    assert r.trace["trace_missing"] == ["debug"]


def test_trace_path_invalid_type_rejected():
    with pytest.raises(ValueError, match="trace_path"):
        HTTPAdapter(URL, trace_path=["data"])
