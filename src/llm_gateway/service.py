"""应用服务层：组装各模块，并实现四个业务用例（chat / extract / tools / stream）。

分层（对应文章「推荐的分层」）：
    HTTP 路由（api.py）
      └─ 应用服务（本文件）：输入校验、会话策略、预算与业务规则
           ├─ 模型适配器 + 网关：认证、请求映射、重试、响应归一化、日志指标
           ├─ 工具注册表：参数校验、授权、执行、审计
           └─ 状态存储：会话（SessionStore）与业务数据（OrderStore）
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from .adapters.mock_server import MockLLMServer
from .adapters.openai_compat import OpenAICompatAdapter
from .config import Settings
from .context import BuiltContext, ContextBuilder, Session, SessionStore
from .errors import ErrorCategory, GatewayError
from .extraction import ExtractionResult, TicketExtractor
from .gateway import ModelGateway
from .models import Message, ModelRequest, StreamEnd, StreamEvent, StreamStart, TextDelta
from .observability import Metrics, log_event
from .prompting import PromptRegistry, PromptTemplate
from .retry import RetryPolicy
from .sse import format_sse
from .tool_loop import ToolLoopResult, run_tool_loop
from .tools import OrderStore, ToolContext, ToolRegistry, get_order_status_tool

# /tools 用例只允许这一个只读工具：只注册本轮任务需要的工具
TOOLS_ALLOWLIST = {"get_order_status"}


@dataclass
class StreamPlan:
    """在返回 200 之前就准备好的流式请求。

    会话、输入长度、上下文预算，以及“建立上游流”（含重试）都在这一步完成：
    这些阶段的错误仍可以用正常的 HTTP 状态码（413、503……）返回给客户端；
    一旦开始输出，之后的错误只能以 SSE 的 error 事件告知。
    """

    session: Session
    context: BuiltContext
    template: PromptTemplate
    user_text: str
    upstream: AsyncIterator[StreamEvent]
    first_event: StreamEvent


class GatewayService:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self.metrics = Metrics(settings.price_input_per_1m, settings.price_output_per_1m)

        if settings.llm_provider == "mock":
            transport = transport or MockLLMServer().transport()
        elif settings.llm_provider != "openai":
            raise ValueError(f"unsupported LLM_PROVIDER: {settings.llm_provider}")

        adapter = OpenAICompatAdapter(settings, transport)
        policy = RetryPolicy(settings.llm_max_retries, settings.llm_retry_base_delay, settings.llm_retry_max_delay)
        self.gateway = ModelGateway(adapter, policy, self.metrics)
        self.prompts = PromptRegistry(settings.prompt_dir)
        self.sessions = SessionStore(settings.session_ttl_seconds)
        self.context = ContextBuilder(settings, self.gateway, self.prompts)
        self.orders = OrderStore()
        self.registry = ToolRegistry(self.metrics, settings.tool_timeout_seconds, settings.tool_result_max_chars)
        self.registry.register(get_order_status_tool(self.orders))
        self.extractor = TicketExtractor(self.gateway, self.prompts, self.orders, self.metrics)

    async def aclose(self) -> None:
        await self.gateway.adapter.aclose()

    def _check_input(self, text: str) -> str:
        text = text.strip()
        if not text:
            raise GatewayError(ErrorCategory.BAD_INPUT, "message must not be empty")
        if len(text) > self.settings.max_input_chars:
            raise GatewayError(ErrorCategory.INPUT_TOO_LONG, f"message exceeds {self.settings.max_input_chars} chars")
        return text

    # ------------------------------------------------------------------ /chat

    async def chat(self, user_id: str, conversation_id: str | None, text: str) -> dict[str, Any]:
        text = self._check_input(text)
        session = self.sessions.get_or_create(user_id, conversation_id)
        template = self.prompts.get("chat")
        ctx = await self.context.build(session, template.render(), text)
        resp = await self.gateway.generate(ModelRequest(ctx.messages, operation="chat"), prompt_ref=template.ref)

        if resp.refusal:
            status, reply = "refused", resp.refusal
        elif resp.finish_reason == "length":
            status, reply = "truncated", resp.text or ""
        elif resp.finish_reason == "content_filter":
            status, reply = "blocked", ""
        elif not resp.text:
            status, reply = "empty", ""
        else:
            status, reply = "ok", resp.text
            # 只有完整成功的一轮才写入历史
            self.sessions.save_turn(session, text, reply)

        return {
            "conversation_id": session.conversation_id,
            "status": status,
            "reply": reply,
            "finish_reason": resp.finish_reason,
            "model": resp.model,
            "prompt_ref": template.ref,
            "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
            "attempts": resp.attempts,
            "provider_request_id": resp.provider_request_id,
            "context": _context_info(ctx),
        }

    # ------------------------------------------------------------------ /extract

    async def extract(self, text: str) -> ExtractionResult:
        return await self.extractor.extract(self._check_input(text))

    # ------------------------------------------------------------------ /tools

    async def run_tools(self, user_id: str, text: str) -> ToolLoopResult:
        text = self._check_input(text)
        template = self.prompts.get("tools")
        # 系统规则来自模板，用户输入作为独立的 user 消息，二者不拼接在一起
        messages = template.render() + [Message("user", text)]
        return await run_tool_loop(
            self.gateway,
            self.registry,
            messages,
            TOOLS_ALLOWLIST,
            ToolContext(user_id=user_id),
            max_rounds=self.settings.tool_max_rounds,
            total_timeout=self.settings.tool_total_timeout_seconds,
            prompt_ref=template.ref,
        )

    # ------------------------------------------------------------------ /stream

    async def prepare_stream(self, user_id: str, conversation_id: str | None, text: str) -> StreamPlan:
        text = self._check_input(text)
        session = self.sessions.get_or_create(user_id, conversation_id)
        template = self.prompts.get("chat")
        ctx = await self.context.build(session, template.render(), text)
        upstream = self.gateway.stream(ModelRequest(ctx.messages, operation="stream"), prompt_ref=template.ref)
        first = await anext(upstream)  # 建流失败时异常在这里抛出，由路由层映射为 HTTP 错误
        return StreamPlan(session, ctx, template, text, upstream, first)

    async def stream(
        self, plan: StreamPlan, is_disconnected: Callable[[], Awaitable[bool]]
    ) -> AsyncIterator[bytes]:
        """把统一流式事件转换成发给浏览器的 SSE：start → delta* → done | error。

        - 文本边到边转发，但只有收到 done 才代表结果完整；
        - 中途出错时发送 error 事件，并通过 partial 字段说明是否已经显示过部分内容；
        - 客户端断开时停止读取上游并释放连接（gateway.stream 的 finally 会关闭上游）。
        """
        conv_id = plan.session.conversation_id
        upstream = plan.upstream
        parts: list[str] = []

        events = _prepend(plan.first_event, upstream)  # 把建流时已取出的首个事件放回序列
        try:
            async for event in events:
                if await is_disconnected():
                    log_event("stream_client_disconnected")
                    break
                if isinstance(event, StreamStart):
                    yield format_sse("start", {"conversation_id": conv_id, "model": event.model,
                                               "prompt_ref": plan.template.ref})
                elif isinstance(event, TextDelta):
                    parts.append(event.text)
                    yield format_sse("delta", {"text": event.text})
                elif isinstance(event, StreamEnd):
                    text = "".join(parts)
                    complete = event.finish_reason == "stop" and bool(text)
                    if complete:
                        self.sessions.save_turn(plan.session, plan.user_text, text)
                    usage = event.usage
                    yield format_sse("done", {
                        "conversation_id": conv_id,
                        "finish_reason": event.finish_reason,
                        "complete": complete,
                        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
                        if usage else None,
                    })
                # ToolCallDelta：本接口不开放工具，收到也不转发
        except GatewayError as err:
            yield format_sse("error", {**err.to_dict(), "partial": bool(parts)})
        finally:
            await events.aclose()  # type: ignore[attr-defined]
            await upstream.aclose()  # type: ignore[attr-defined]  # 关闭上游连接，网关记录 cancelled


async def _prepend(first: StreamEvent, rest: AsyncIterator[StreamEvent]) -> AsyncIterator[StreamEvent]:
    yield first
    async for event in rest:
        yield event


def _context_info(ctx: BuiltContext) -> dict[str, Any]:
    return {
        "estimated_input_tokens": ctx.estimated_input_tokens,
        "history_messages": ctx.history_messages,
        "has_summary": ctx.has_summary,
        "summarized_messages": ctx.summarized_messages,
        "dropped_messages": ctx.dropped_messages,
    }
