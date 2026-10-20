"""错误分类。

对应文章「错误处理与可靠性」：先按“是否值得重试”和“重试是否有副作用”给错误分类，
而不是捕获所有异常后一律重试。每个类别决定：是否重试、返回给客户端的 HTTP 状态、日志里的错误类别。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import httpx


class ErrorCategory(StrEnum):
    INVALID_REQUEST = "invalid_request"  # 字段错误、参数不支持 → 修代码，不重试
    AUTH = "auth"  # 上游 Key 错误或无权限 → 停止，检查配置
    RATE_LIMITED = "rate_limited"  # 429 限流 → 退避重试
    QUOTA_EXCEEDED = "quota_exceeded"  # 额度耗尽 → 重试无用
    SERVER_ERROR = "server_error"  # 5xx → 有界重试
    TIMEOUT = "timeout"  # 连接/读取超时 → 有界重试
    NETWORK = "network"  # DNS、连接断开 → 有界重试
    MODEL_OUTPUT_UNUSABLE = "model_output_unusable"  # 拒绝、截断、Schema 不符 → 不自动重试
    TOOL_FAILED = "tool_failed"  # 工具执行失败
    TOOL_LOOP_LIMIT = "tool_loop_limit"  # 工具循环超出轮数/时间预算
    # 以下是网关自身对客户端的错误
    UNAUTHORIZED = "unauthorized"
    BAD_INPUT = "bad_input"
    INPUT_TOO_LONG = "input_too_long"


RETRYABLE = {
    ErrorCategory.RATE_LIMITED,
    ErrorCategory.SERVER_ERROR,
    ErrorCategory.TIMEOUT,
    ErrorCategory.NETWORK,
}

# 网关返回给客户端的 HTTP 状态。上游的认证/参数错误属于网关自身配置问题，
# 对客户端统一表现为 502，避免泄露上游细节。
HTTP_STATUS = {
    ErrorCategory.INVALID_REQUEST: 502,
    ErrorCategory.AUTH: 502,
    ErrorCategory.RATE_LIMITED: 503,
    ErrorCategory.QUOTA_EXCEEDED: 503,
    ErrorCategory.SERVER_ERROR: 502,
    ErrorCategory.TIMEOUT: 504,
    ErrorCategory.NETWORK: 502,
    ErrorCategory.MODEL_OUTPUT_UNUSABLE: 422,
    ErrorCategory.TOOL_FAILED: 502,
    ErrorCategory.TOOL_LOOP_LIMIT: 422,
    ErrorCategory.UNAUTHORIZED: 401,
    ErrorCategory.BAD_INPUT: 400,
    ErrorCategory.INPUT_TOO_LONG: 413,
}


class GatewayError(Exception):
    """网关内统一的异常类型，携带分类和排查信息。"""

    def __init__(
        self,
        category: ErrorCategory,
        message: str,
        *,
        retry_after: float | None = None,
        provider_request_id: str | None = None,
        upstream_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.retry_after = retry_after
        self.provider_request_id = provider_request_id
        self.upstream_status = upstream_status
        self.attempts = 1  # 由重试层更新为实际尝试次数

    @property
    def retryable(self) -> bool:
        return self.category in RETRYABLE

    @property
    def http_status(self) -> int:
        return HTTP_STATUS[self.category]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "message": self.message,
            "retryable": self.retryable,
        }


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    """读取服务端给出的重试提示（Retry-After 秒数，或部分服务的毫秒头）。"""
    if ms := headers.get("retry-after-ms"):
        try:
            return float(ms) / 1000
        except ValueError:
            pass
    if sec := headers.get("retry-after"):
        try:
            return float(sec)
        except ValueError:
            return None  # HTTP-date 格式在此示例中忽略，交给指数退避
    return None


def classify_http_error(response: httpx.Response, body: Any) -> GatewayError:
    """把上游的非 2xx 响应映射为错误类别。错误码随供应商变化，需定期对照官方文档更新。"""
    status = response.status_code
    request_id = response.headers.get("x-request-id")
    err = body.get("error", {}) if isinstance(body, dict) else {}
    code = str(err.get("code") or err.get("type") or "")
    message = str(err.get("message") or f"upstream HTTP {status}")

    if status == 429:
        # 同样是 429，“额度耗尽”重试没有意义，“限流”才值得退避重试
        category = (
            ErrorCategory.QUOTA_EXCEEDED
            if "quota" in code or "insufficient" in code
            else ErrorCategory.RATE_LIMITED
        )
    elif status in (401, 403):
        category = ErrorCategory.AUTH
    elif status in (408,):
        category = ErrorCategory.TIMEOUT
    elif 400 <= status < 500:
        category = ErrorCategory.INVALID_REQUEST
    else:
        category = ErrorCategory.SERVER_ERROR

    return GatewayError(
        category,
        message,
        retry_after=_parse_retry_after(response.headers),
        provider_request_id=request_id,
        upstream_status=status,
    )


def classify_transport_error(exc: httpx.TransportError) -> GatewayError:
    """区分连接超时、读取超时和其他网络错误。"""
    if isinstance(exc, httpx.ConnectTimeout):
        return GatewayError(ErrorCategory.TIMEOUT, "connect timeout")
    if isinstance(exc, httpx.TimeoutException):
        return GatewayError(ErrorCategory.TIMEOUT, f"{type(exc).__name__}")
    return GatewayError(ErrorCategory.NETWORK, f"{type(exc).__name__}")
