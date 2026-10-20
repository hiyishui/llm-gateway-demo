"""HTTP 路由层：认证、trace ID、请求体校验、错误到 HTTP 状态的映射。

接口一览：
    POST   /chat                    多轮对话（会话隔离 + 上下文预算）
    POST   /extract                 工单结构化抽取（三层校验）
    POST   /tools                   只读工具调用循环
    POST   /stream                  流式对话（SSE）
    DELETE /sessions/{conversation_id}  删除会话
    GET    /metrics                 延迟、Token、成本、成功率
    GET    /healthz                 健康检查（无需认证）
"""

from __future__ import annotations

import math
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .errors import ErrorCategory, GatewayError
from .observability import hash_user, log_event, setup_logging, trace_id_var
from .service import GatewayService


class ChatBody(BaseModel):
    message: str = Field(..., description="用户输入")
    conversation_id: str | None = Field(None, max_length=64, description="会话 ID，省略时新建会话")


class ExtractBody(BaseModel):
    ticket_text: str = Field(..., description="待抽取的工单原文")


class ToolsBody(BaseModel):
    message: str


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log_event("startup", provider=settings.llm_provider, model=settings.llm_model,
                  base_url=settings.llm_base_url if settings.llm_provider != "mock" else "mock")
        yield
        await service.aclose()

    service = GatewayService(settings, transport)
    app = FastAPI(title="LLM Gateway Demo", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        # 每个请求一个 trace ID：可由调用方传入，也可由网关生成；随响应头返回，便于用户报障
        trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex[:16]
        token = trace_id_var.set(trace_id)
        try:
            response = await call_next(request)
        finally:
            trace_id_var.reset(token)
        response.headers["x-trace-id"] = trace_id
        return response

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, err: GatewayError) -> JSONResponse:
        log_event("request_failed", level=40 if err.http_status >= 500 else 30, path=request.url.path,
                  error_category=err.category.value, upstream_status=err.upstream_status)
        headers = {}
        if err.category is ErrorCategory.RATE_LIMITED:
            headers["Retry-After"] = str(max(1, math.ceil(err.retry_after or 1)))
        return JSONResponse(
            status_code=err.http_status,
            content={"error": {**err.to_dict(), "trace_id": trace_id_var.get()}},
            headers=headers,
        )

    def current_user(authorization: str | None = Header(None)) -> str:
        """网关自己的认证：浏览器只拿网关颁发的令牌，永远接触不到模型 API Key。"""
        token = (authorization or "").removeprefix("Bearer ").strip()
        user = settings.gateway_tokens.get(token)
        if not user:
            raise GatewayError(ErrorCategory.UNAUTHORIZED, "missing or invalid gateway token")
        return user

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "provider": settings.llm_provider, "model": settings.llm_model}

    @app.post("/chat")
    async def chat(body: ChatBody, user: str = Depends(current_user)) -> dict:
        log_event("chat_request", user=hash_user(user))
        result = await service.chat(user, body.conversation_id, body.message)
        return {**result, "trace_id": trace_id_var.get()}

    @app.post("/extract")
    async def extract(body: ExtractBody, user: str = Depends(current_user)) -> dict:
        result = await service.extract(body.ticket_text)
        return {**result.to_dict(), "trace_id": trace_id_var.get()}

    @app.post("/tools")
    async def tools(body: ToolsBody, user: str = Depends(current_user)) -> dict:
        result = await service.run_tools(user, body.message)
        return {**result.to_dict(), "trace_id": trace_id_var.get()}

    @app.post("/stream")
    async def stream(body: ChatBody, request: Request, user: str = Depends(current_user)) -> StreamingResponse:
        # 输入、上下文预算和“建立上游流（含重试）”都在开始输出之前完成，
        # 这些阶段的错误仍可返回正常的 4xx/5xx；开始输出后的错误以 SSE error 事件告知
        plan = await service.prepare_stream(user, body.conversation_id, body.message)
        return StreamingResponse(
            service.stream(plan, request.is_disconnected),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/sessions/{conversation_id}")
    async def delete_session(conversation_id: str, user: str = Depends(current_user)) -> dict:
        return {"deleted": service.sessions.delete(user, conversation_id)}

    @app.get("/metrics")
    async def metrics(user: str = Depends(current_user)) -> dict:
        return service.metrics.snapshot()

    return app
