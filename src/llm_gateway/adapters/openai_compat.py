"""OpenAI 兼容的 Chat Completions 适配器。

对应文章「模型 API」与「流式输出」：
- 负责认证、请求映射、响应归一化、错误分类，以及把 SSE 分片翻译成统一流式事件；
- 通过 LLM_BASE_URL 可切换 OpenAI、DeepSeek、通义千问兼容模式、本地 Ollama 等服务；
- 各家对字段支持程度不同（如 response_format、stream_options），差异通过配置开关处理。

字段格式参考 OpenAI Chat Completions API；接入其他服务时请以其最新文档为准。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..config import Settings
from ..errors import ErrorCategory, GatewayError, classify_http_error, classify_transport_error
from ..models import (
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from ..sse import iter_sse

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}


class OpenAICompatAdapter:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self.model = settings.llm_model
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            # 密钥只在这里使用一次，之后不再以明文出现在任何日志或异常信息里
            headers["Authorization"] = f"Bearer {settings.llm_api_key.reveal()}"
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            headers=headers,
            # 区分连接超时与读取超时：连不上要快速失败；生成内容可能需要更久
            timeout=httpx.Timeout(
                connect=settings.llm_connect_timeout,
                read=settings.llm_read_timeout,
                write=10.0,
                pool=5.0,
            ),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ 请求映射

    def build_payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._map_message(m) for m in request.messages],
            self.settings.llm_max_tokens_param: request.max_output_tokens
            or self.settings.llm_max_output_tokens,
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in request.tools
            ]
        if spec := request.structured_output:
            mode = self.settings.llm_structured_output_mode
            if mode == "json_schema":
                # 由服务端按 Schema 约束生成；strict 要求 Schema 中所有字段都 required
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": spec.name, "schema": spec.schema, "strict": True},
                }
            elif mode == "json_object":
                # 只保证是合法 JSON，字段约束靠 Prompt 中附带的 Schema + 本地校验
                payload["response_format"] = {"type": "json_object"}
            # mode == "prompt"：不发送 response_format，完全依赖 Prompt 与本地校验
        if stream:
            payload["stream"] = True
            if self.settings.llm_stream_include_usage:
                payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _map_message(m: Message) -> dict[str, Any]:
        if m.role == "tool":
            # 工具结果必须带上 tool_call_id，模型才能把结果与它发起的调用对应起来
            return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""}
        out: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            out["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments_json}}
                for c in m.tool_calls
            ]
        return out

    # ------------------------------------------------------------------ 非流式调用

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload = self.build_payload(request, stream=False)
        started = time.perf_counter()
        try:
            resp = await self._client.post("/chat/completions", json=payload)
        except httpx.TransportError as exc:
            raise classify_transport_error(exc) from exc

        body = _safe_json(resp)
        if resp.status_code >= 400:
            raise classify_http_error(resp, body)
        if not isinstance(body, dict) or not body.get("choices"):
            # 200 但内容无法解析：视为上游临时故障
            raise GatewayError(
                ErrorCategory.SERVER_ERROR,
                "upstream returned an unparsable body",
                provider_request_id=resp.headers.get("x-request-id"),
            )

        choice = body["choices"][0]
        msg = choice.get("message") or {}
        tool_calls = [
            ToolCall(
                id=c.get("id", ""),
                name=(c.get("function") or {}).get("name", ""),
                arguments_json=(c.get("function") or {}).get("arguments") or "{}",
            )
            for c in msg.get("tool_calls") or []
        ]
        finish = _FINISH_REASONS.get(choice.get("finish_reason") or "", "unknown")
        if tool_calls:
            finish = "tool_calls"  # 部分兼容服务在有工具调用时仍返回 stop
        usage = body.get("usage") or {}
        return ModelResponse(
            text=msg.get("content"),
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
            model=body.get("model", self.model),
            provider_request_id=resp.headers.get("x-request-id") or body.get("id"),
            refusal=msg.get("refusal"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # ------------------------------------------------------------------ 流式调用

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """把上游 SSE 翻译成统一事件。

        - 建立连接前后的 HTTP 错误按状态码分类；
        - 连接建立（200）之后，错误也可能以流内事件出现，必须逐个检查；
        - 消费方提前 aclose() 这个生成器时，async with 会关闭上游连接，及时释放资源。
        """
        payload = self.build_payload(request, stream=True)
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise classify_http_error(resp, _safe_json(resp))

                request_id = resp.headers.get("x-request-id")
                yield StreamStart(model=self.model, provider_request_id=request_id)

                finish: FinishReason | None = None
                usage: Usage | None = None
                async for sse in iter_sse(resp.aiter_bytes()):
                    if sse.data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(sse.data)
                    except json.JSONDecodeError as exc:
                        raise GatewayError(ErrorCategory.SERVER_ERROR, "malformed stream chunk") from exc

                    if err := chunk.get("error"):
                        # 流内错误：HTTP 状态码早已是 200，只能从事件内容中识别
                        raise GatewayError(
                            ErrorCategory.SERVER_ERROR,
                            str(err.get("message", "stream error")),
                            provider_request_id=request_id,
                        )
                    if u := chunk.get("usage"):
                        usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if text := delta.get("content"):
                            yield TextDelta(text)
                        for tc in delta.get("tool_calls") or []:
                            fn = tc.get("function") or {}
                            yield ToolCallDelta(
                                index=tc.get("index", 0),
                                id=tc.get("id"),
                                name=fn.get("name"),
                                arguments_fragment=fn.get("arguments") or "",
                            )
                        if reason := choice.get("finish_reason"):
                            finish = _FINISH_REASONS.get(reason, "unknown")

                if finish is None:
                    # 连接正常关闭却没有结束原因：说明输出不完整，不能当作成功
                    raise GatewayError(ErrorCategory.NETWORK, "stream ended without finish_reason")
                yield StreamEnd(finish_reason=finish, usage=usage)
        except httpx.TransportError as exc:
            raise classify_transport_error(exc) from exc


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
