import pytest

from ai_eval.adapters.chat_adapter import AnthropicAdapter, OpenAIAdapter
from ai_eval.llm import LLMError, backoff_delay, call_with_retries, chat, is_retryable
from ai_eval.models import EvalCase, RunResult
from ai_eval.scorers.llm_judge import LLMJudgeScorer
from fakes import FakeAnthropic, FakeAPIError, FakeOpenAI

CASE = EvalCase(id="1", input="q")


class APITimeoutError(Exception):  # same class name as the openai/anthropic SDK timeout
    pass


class APIConnectionError(Exception):
    pass


class Flaky:
    """Callable that raises the queued errors, then returns "done"."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "done"


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 529])
def test_transient_status_codes_are_retryable(status):
    assert is_retryable(FakeAPIError(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(status):
    assert not is_retryable(FakeAPIError(status))


@pytest.mark.parametrize(
    "exc", [TimeoutError(), ConnectionError(), APITimeoutError(), APIConnectionError()]
)
def test_timeouts_and_connection_errors_are_retryable(exc):
    assert is_retryable(exc)


@pytest.mark.parametrize("exc", [RuntimeError("boom"), ValueError(), LLMError("OPENAI_API_KEY is not set")])
def test_other_errors_are_not_retryable(exc):
    assert not is_retryable(exc)


def test_backoff_doubles_with_jitter_and_is_capped():
    assert backoff_delay(0, 0.5, 20.0, rng=lambda: 1.0) == 0.5
    assert backoff_delay(3, 0.5, 20.0, rng=lambda: 1.0) == 4.0
    assert backoff_delay(3, 0.5, 20.0, rng=lambda: 0.0) == 2.0  # jitter floor is 50%
    assert backoff_delay(10, 0.5, 20.0, rng=lambda: 1.0) == 20.0


def test_fails_then_succeeds():
    sleeps = []
    fn = Flaky(FakeAPIError(429), FakeAPIError(503))
    result, retries = call_with_retries(fn, max_retries=3, base_delay=1.0, sleep=sleeps.append)
    assert (result, retries, fn.calls) == ("done", 2, 3)
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 2.0  # exponential, jittered


def test_gives_up_after_max_retries_and_raises_last_error():
    sleeps = []
    last = FakeAPIError(500)
    fn = Flaky(FakeAPIError(500), FakeAPIError(500), last)
    with pytest.raises(FakeAPIError) as info:
        call_with_retries(fn, max_retries=2, sleep=sleeps.append)
    assert info.value is last
    assert fn.calls == 3 and len(sleeps) == 2


def test_non_retryable_error_fails_immediately():
    sleeps = []
    fn = Flaky(FakeAPIError(401))
    with pytest.raises(FakeAPIError):
        call_with_retries(fn, max_retries=3, sleep=sleeps.append)
    assert fn.calls == 1 and sleeps == []


def test_max_retries_zero_disables_retrying():
    fn = Flaky(FakeAPIError(429))
    with pytest.raises(FakeAPIError):
        call_with_retries(fn, max_retries=0, sleep=lambda s: pytest.fail("should not sleep"))
    assert fn.calls == 1


def test_retry_after_header_is_honoured_and_capped():
    sleeps = []
    fn = Flaky(FakeAPIError(429, retry_after=7), FakeAPIError(429, retry_after=600))
    call_with_retries(fn, max_retries=3, max_delay=20.0, sleep=sleeps.append)
    assert sleeps == [7.0, 20.0]


@pytest.mark.parametrize(
    ("fake", "provider"),
    [(FakeOpenAI, "openai"), (FakeAnthropic, "anthropic")],
)
def test_chat_retries_each_provider_and_reports_count(fake, provider):
    sleeps = []
    client = fake(text="hi", errors=[FakeAPIError(429), APITimeoutError()])
    resp = chat(client, provider, "m", "sys", "user", sleep=sleeps.append)
    assert resp.text == "hi" and resp.retries == 2
    assert len(client.calls) == 3 and len(sleeps) == 2


@pytest.mark.parametrize("adapter_cls", [OpenAIAdapter, AnthropicAdapter])
def test_adapter_records_retries_in_trace(adapter_cls):
    fake = FakeOpenAI if adapter_cls is OpenAIAdapter else FakeAnthropic
    client = fake(text="answer", errors=[FakeAPIError(503, retry_after=0)])
    run = adapter_cls(client=client).run(CASE)
    assert run.error is None and run.output == "answer"
    assert run.trace["retries"] == 1
    assert len(client.calls) == 2


def test_adapter_reports_error_once_retries_are_exhausted():
    client = FakeOpenAI(errors=[FakeAPIError(500, retry_after=0)] * 2)
    run = OpenAIAdapter(client=client, max_retries=1).run(CASE)
    assert run.output == "" and "FakeAPIError" in run.error
    assert len(client.calls) == 2


def test_negative_max_retries_is_rejected():
    with pytest.raises(ValueError):
        OpenAIAdapter(client=FakeOpenAI(), max_retries=-1)
    with pytest.raises(ValueError):
        LLMJudgeScorer(client=FakeOpenAI(), max_retries=-1)


def test_judge_retries_transient_errors():
    client = FakeOpenAI('{"pass": true, "score": 0.9}', errors=[FakeAPIError(429, retry_after=0)])
    result = LLMJudgeScorer(client=client).score(CASE, RunResult(case_id="1", output="a"))
    assert result.passed and result.score == 0.9
    assert len(client.calls) == 2


def test_judge_with_retries_disabled_fails_on_first_transient_error():
    client = FakeOpenAI('{"pass": true}', errors=[FakeAPIError(429, retry_after=0)])
    result = LLMJudgeScorer(client=client, max_retries=0).score(CASE, RunResult(case_id="1", output="a"))
    assert not result.passed and "FakeAPIError" in result.details["error"]
    assert len(client.calls) == 1
