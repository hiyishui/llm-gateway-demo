"""有界的指数退避重试。

对应文章「错误处理与可靠性 · 有界的指数退避」：
    delay = min(max_delay, base × 2^attempt) + jitter
- 只重试明确可恢复的错误（限流、5xx、超时、网络）；
- 优先遵守服务端的 Retry-After 提示；
- 设最大重试次数，超过后把最后一次错误抛给上层降级处理。
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from .errors import GatewayError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3  # 不含首次尝试；总尝试次数 = 1 + max_retries
    base_delay: float = 0.5
    max_delay: float = 8.0

    def backoff(self, attempt: int, retry_after: float | None) -> float:
        """计算第 attempt 次失败后的等待秒数（attempt 从 0 开始）。"""
        if retry_after is not None:
            # 服务端明确告知等待时间时照做，但仍受 max_delay 约束，避免请求无限挂起
            return min(self.max_delay, retry_after)
        exp = min(self.max_delay, self.base_delay * 2**attempt)
        # 随机抖动：避免大量客户端在同一时刻一起重试，形成“重试风暴”
        return exp + random.uniform(0, self.base_delay)


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    on_retry: Callable[[int, GatewayError, float], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[T, int]:
    """执行 fn，遇到可重试错误时退避重试。返回 (结果, 实际尝试次数)。

    注意：这里只用于“没有副作用”的模型调用。会产生副作用的操作（扣款、发信等）
    需要先有幂等键或任务状态，才能安全重试。
    """
    attempt = 0
    while True:
        try:
            return await fn(), attempt + 1
        except GatewayError as err:
            if not err.retryable or attempt >= policy.max_retries:
                err.attempts = attempt + 1
                raise
            delay = policy.backoff(attempt, err.retry_after)
            if on_retry:
                on_retry(attempt + 1, err, delay)
            await sleep(delay)
            attempt += 1
