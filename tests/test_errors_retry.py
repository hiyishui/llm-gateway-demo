"""错误分类与有界重试。"""

import httpx
import pytest

from llm_gateway.errors import ErrorCategory, GatewayError, classify_http_error, classify_transport_error
from llm_gateway.retry import RetryPolicy, call_with_retry


def _resp(status: int, code: str = "", headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": "m"}}, headers=headers or {})


@pytest.mark.parametrize(
    ("status", "code", "category", "retryable"),
    [
        (400, "invalid_request_error", ErrorCategory.INVALID_REQUEST, False),
        (401, "invalid_api_key", ErrorCategory.AUTH, False),
        (429, "rate_limit_exceeded", ErrorCategory.RATE_LIMITED, True),
        (429, "insufficient_quota", ErrorCategory.QUOTA_EXCEEDED, False),
        (500, "server_error", ErrorCategory.SERVER_ERROR, True),
        (503, "", ErrorCategory.SERVER_ERROR, True),
    ],
)
def test_classify_http_error(status, code, category, retryable):
    err = classify_http_error(_resp(status, code), _resp(status, code).json())
    assert err.category is category
    assert err.retryable is retryable


def test_retry_after_header_is_parsed():
    resp = _resp(429, "rate_limit_exceeded", {"retry-after": "2"})
    assert classify_http_error(resp, resp.json()).retry_after == 2.0


def test_transport_errors_distinguish_timeout_and_network():
    req = httpx.Request("POST", "http://x")
    assert classify_transport_error(httpx.ConnectTimeout("t", request=req)).category is ErrorCategory.TIMEOUT
    assert classify_transport_error(httpx.ReadTimeout("t", request=req)).category is ErrorCategory.TIMEOUT
    assert classify_transport_error(httpx.ConnectError("c", request=req)).category is ErrorCategory.NETWORK


def test_backoff_is_exponential_bounded_and_respects_retry_after():
    policy = RetryPolicy(max_retries=5, base_delay=1.0, max_delay=4.0)
    assert 1.0 <= policy.backoff(0, None) <= 2.0
    assert 2.0 <= policy.backoff(1, None) <= 3.0
    assert 4.0 <= policy.backoff(5, None) <= 5.0  # 指数部分被 max_delay 封顶，只多出抖动
    assert policy.backoff(0, 3.0) == 3.0  # 服务端提示优先
    assert policy.backoff(0, 60.0) == 4.0  # 但不超过 max_delay


async def _no_sleep(_: float) -> None:
    return None


async def test_retryable_error_recovers():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise GatewayError(ErrorCategory.SERVER_ERROR, "boom")
        return "ok"

    result, attempts = await call_with_retry(flaky, RetryPolicy(max_retries=3), sleep=_no_sleep)
    assert (result, attempts) == ("ok", 3)


async def test_non_retryable_error_is_not_retried():
    calls = {"n": 0}

    async def bad():
        calls["n"] += 1
        raise GatewayError(ErrorCategory.INVALID_REQUEST, "bad")

    with pytest.raises(GatewayError):
        await call_with_retry(bad, RetryPolicy(max_retries=3), sleep=_no_sleep)
    assert calls["n"] == 1


async def test_retry_gives_up_after_max_attempts():
    async def always_429():
        raise GatewayError(ErrorCategory.RATE_LIMITED, "slow down")

    with pytest.raises(GatewayError) as info:
        await call_with_retry(always_429, RetryPolicy(max_retries=2), sleep=_no_sleep)
    assert info.value.attempts == 3
