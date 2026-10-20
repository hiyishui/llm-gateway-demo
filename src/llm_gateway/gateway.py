"""模型网关：在适配器之外统一加上重试、日志与指标。

业务代码通过 ModelGateway 调用模型，不直接接触适配器，因此每一次调用——无论成功、失败
还是被客户端取消——都会留下同样口径的诊断记录。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from .adapters.base import ModelAdapter
from .errors import GatewayError
from .models import ModelRequest, ModelResponse, StreamEnd, StreamEvent, StreamStart, TextDelta
from .observability import CallRecord, Metrics, log_event
from .retry import RetryPolicy, call_with_retry
from .tokens import estimate_messages, estimate_text


class ModelGateway:
    def __init__(self, adapter: ModelAdapter, policy: RetryPolicy, metrics: Metrics) -> None:
        self.adapter = adapter
        self.policy = policy
        self.metrics = metrics

    @property
    def model(self) -> str:
        return self.adapter.model

    def _on_retry(self, operation: str):
        def log(attempt: int, err: GatewayError, delay: float) -> None:
            log_event(
                "model_retry",
                level=logging.WARNING,
                operation=operation,
                attempt=attempt,
                error_category=err.category.value,
                upstream_status=err.upstream_status,
                delay_s=round(delay, 2),
            )

        return log

    async def generate(self, request: ModelRequest, prompt_ref: str | None = None) -> ModelResponse:
        rec = CallRecord(operation=request.operation, model=self.model, prompt_ref=prompt_ref)
        started = time.perf_counter()
        try:
            resp, attempts = await call_with_retry(
                lambda: self.adapter.generate(request), self.policy, on_retry=self._on_retry(request.operation)
            )
        except GatewayError as err:
            rec.status, rec.error_category = "error", err.category.value
            rec.attempts, rec.provider_request_id = err.attempts, err.provider_request_id
            raise
        finally:
            rec.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            if rec.status != "ok":
                self.metrics.record_call(rec)

        resp.attempts = attempts
        rec.attempts = attempts
        rec.provider_request_id = resp.provider_request_id
        rec.model = resp.model
        rec.finish_reason = resp.finish_reason
        rec.input_tokens, rec.output_tokens = resp.usage.input_tokens, resp.usage.output_tokens
        rec.tool_calls = [c.name for c in resp.tool_calls]
        self.metrics.record_call(rec)
        return resp

    async def stream(self, request: ModelRequest, prompt_ref: str | None = None) -> AsyncIterator[StreamEvent]:
        """流式调用。

        重试只发生在“建立流”阶段（拿到第一个事件之前）；一旦开始向用户输出内容，
        再重试就会重复显示文字，所以之后的错误直接上抛，由上层告知用户结果不完整。
        """
        rec = CallRecord(operation=request.operation, model=self.model, prompt_ref=prompt_ref)
        started = time.perf_counter()
        upstream: AsyncIterator[StreamEvent] | None = None
        output_chars: list[str] = []

        async def open_stream() -> StreamEvent:
            nonlocal upstream
            upstream = self.adapter.stream(request)
            try:
                return await anext(upstream)
            except BaseException:
                await upstream.aclose()  # type: ignore[attr-defined]
                raise

        try:
            first, attempts = await call_with_retry(open_stream, self.policy, on_retry=self._on_retry("stream"))
            rec.attempts = attempts
            if isinstance(first, StreamStart):
                rec.provider_request_id = first.provider_request_id
            yield first
            assert upstream is not None
            async for event in upstream:
                if isinstance(event, TextDelta):
                    output_chars.append(event.text)
                elif isinstance(event, StreamEnd):
                    rec.finish_reason = event.finish_reason
                    if event.usage:
                        rec.input_tokens = event.usage.input_tokens
                        rec.output_tokens = event.usage.output_tokens
                yield event
        except GatewayError as err:
            rec.status, rec.error_category = "error", err.category.value
            rec.attempts = max(rec.attempts, err.attempts)
            raise
        except (asyncio.CancelledError, GeneratorExit):
            # 客户端断开或主动取消：记录为 cancelled，而不是成功
            rec.status = "cancelled"
            raise
        finally:
            if upstream is not None:
                await upstream.aclose()  # type: ignore[attr-defined]  # 释放上游连接
            if not rec.output_tokens:
                # 服务未返回流式 usage 时，用估算值兜底（日志中可据此区分）
                rec.input_tokens = rec.input_tokens or estimate_messages(request.messages)
                rec.output_tokens = estimate_text("".join(output_chars))
            rec.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            self.metrics.record_call(rec)
