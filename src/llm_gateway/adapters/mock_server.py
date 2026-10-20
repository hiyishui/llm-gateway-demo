"""离线 Mock 模型服务（LLM_PROVIDER=mock 时使用）。

它不是“假适配器”，而是在 HTTP 层模拟一个 OpenAI 兼容服务：
网关照常走 OpenAICompatAdapter 的请求映射、错误分类与 SSE 解析，只是请求被
httpx.MockTransport 截获，由这里按简单规则“扮演”模型。这样没有 API Key 也能完整
演示和测试所有流程。

为了演示错误处理，用户输入中可以包含故障注入标记：
    [fault:429]  [fault:quota]  [fault:500]  [fault:flaky]  [fault:timeout]  [fault:network]
    [fault:auth] [fault:400]    [fault:refusal] [fault:truncate] [fault:invalid_json]
    [fault:schema] [fault:stream_error] [fault:slow] [fault:unknown_tool] [fault:bad_args]
    [fault:dup]  [fault:loop]
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..tokens import estimate_text

ORDER_ID = re.compile(r"\b([A-Z]\d{3})\b")
FAULT = re.compile(r"\[fault:([a-z_0-9]+)\]")
NAME = re.compile(r"我叫(?!什么)([\u4e00-\u9fa5A-Za-z]{1,10})")


class MockLLMServer:
    def __init__(self, latency_ms: tuple[int, int] = (20, 80), chunk_delay: float = 0.02) -> None:
        self.latency_ms = latency_ms
        self.chunk_delay = chunk_delay
        self._flaky_seen: set[str] = set()

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    # ------------------------------------------------------------------ 入口

    async def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        messages: list[dict[str, Any]] = body["messages"]
        last_user = _latest(messages, "user")
        faults = set(FAULT.findall(last_user))

        if self.latency_ms[1] > 0:
            await asyncio.sleep(random.uniform(*self.latency_ms) / 1000)

        if error := self._injected_error(faults, body, request):
            return error

        if body.get("stream"):
            return self._stream_response(body, faults)
        return self._json_response(body, self._decide(body, messages, last_user, faults))

    # ------------------------------------------------------------------ 故障注入

    def _injected_error(self, faults: set[str], body: dict, request: httpx.Request) -> httpx.Response | None:
        if "timeout" in faults:
            raise httpx.ReadTimeout("mock read timeout", request=request)
        if "network" in faults:
            raise httpx.ConnectError("mock connection refused", request=request)
        if "auth" in faults:
            return _error(401, "Incorrect API key provided", "invalid_api_key")
        if "400" in faults:
            return _error(400, "Unsupported parameter", "invalid_request_error")
        if "429" in faults:
            return _error(429, "Rate limit reached", "rate_limit_exceeded", {"retry-after": "1"})
        if "quota" in faults:
            return _error(429, "You exceeded your current quota", "insufficient_quota")
        if "500" in faults:
            return _error(500, "The server had an error", "server_error")
        if "flaky" in faults:
            # 同一请求第一次返回 503，之后成功：用于演示“重试后恢复”
            key = hashlib.sha256(json.dumps(body["messages"], sort_keys=True).encode()).hexdigest()
            if key not in self._flaky_seen:
                self._flaky_seen.add(key)
                return _error(503, "Service temporarily unavailable", "server_error")
        return None

    # ------------------------------------------------------------------ 决定“模型”的回答

    def _decide(self, body: dict, messages: list[dict], last_user: str, faults: set[str]) -> dict[str, Any]:
        """返回 OpenAI 格式的 message 与 finish_reason。"""
        if "refusal" in faults:
            return {"message": {"role": "assistant", "content": None, "refusal": "抱歉，我无法协助处理该请求。"},
                    "finish_reason": "stop"}
        if body.get("response_format") or "JSON Schema" in _system(messages):
            return self._extract(last_user, faults)
        if body.get("tools"):
            return self._tool_step(body, messages, last_user, faults)
        if "对话摘要" in _system(messages):
            return _text(self._summarize(last_user))
        text = self._chat_reply(messages, last_user)
        if "truncate" in faults:
            return {"message": {"role": "assistant", "content": text[:12]}, "finish_reason": "length"}
        return _text(text)

    def _chat_reply(self, messages: list[dict], last_user: str) -> str:
        users = [m["content"] for m in messages if m["role"] == "user"]
        context = " ".join(users + [_system(messages)])
        if "我叫什么" in last_user:
            names = NAME.findall(context)
            return f"你之前告诉我，你叫{names[-1]}。" if names else "我不知道你的名字，这段对话里没有提到。"
        has_summary = "历史摘要（" in _system(messages)  # 网关注入的摘要消息前缀
        preview = FAULT.sub("", last_user).strip()[:40]
        return (
            f"（mock 回复）收到：「{preview}」。本轮我能看到 {len(users)} 条用户消息"
            f"{'和一段历史摘要' if has_summary else ''}。"
        )

    def _summarize(self, transcript: str) -> str:
        facts = [f"用户说过“我叫{n}”" for n in dict.fromkeys(NAME.findall(transcript))]
        lines = [ln.strip() for ln in transcript.splitlines() if ln.startswith("user:")]
        topics = "；".join(ln.removeprefix("user:").strip()[:20] for ln in lines[-5:])
        return "；".join(facts + [f"用户曾提到：{topics}" if topics else "无其他要点"])

    def _extract(self, prompt: str, faults: set[str]) -> dict[str, Any]:
        """按关键词规则“抽取”工单字段。刻意保持朴素：它会被注入文字影响，
        用来证明真正的防线在应用的 Schema 与业务校验，而不是 Prompt。"""
        m = re.search(r"<ticket>(.*?)</ticket>", prompt, re.S)
        ticket = (m.group(1) if m else prompt).strip()
        if "invalid_json" in faults:
            return _text('{"category": "退款", "order_id": ')
        if "truncate" in faults:
            truncated = '{"category": "物流", "summ'
            return {"message": {"role": "assistant", "content": truncated}, "finish_reason": "length"}
        if "schema" in faults:
            return _text(json.dumps({"category": "投诉", "order_id": None, "summary": "x"}, ensure_ascii=False))

        if any(k in ticket for k in ("退款", "退货", "退钱")):
            category = "退款"
        elif any(k in ticket for k in ("物流", "快递", "没收到", "发货")):
            category = "物流"
        elif any(k in ticket for k in ("坏", "故障", "无法", "不能", "异常")):
            category = "产品问题"
        else:
            category = "其他"
        ids = ORDER_ID.findall(ticket)
        clean = FAULT.sub("", ticket).strip()
        result = {
            "category": category,
            "order_id": ids[0] if ids else None,
            "summary": clean[:60] or "（空工单）",
            "needs_human_review": False,
        }
        return _text(json.dumps(result, ensure_ascii=False))

    def _tool_step(self, body: dict, messages: list[dict], last_user: str, faults: set[str]) -> dict[str, Any]:
        tool_names = {t["function"]["name"] for t in body["tools"]}
        ids = ORDER_ID.findall(last_user)
        # 当前用户问题之后已经产生的工具结果
        idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
        results = [m for m in messages[idx + 1 :] if m["role"] == "tool"]

        if not results or "loop" in faults:
            if "unknown_tool" in faults:
                return _calls([("cancel_order", {"order_id": ids[0] if ids else "A102"})])
            if "bad_args" in faults:
                return _calls([("get_order_status", {"order_id": 102, "force": True})])
            if not ids:
                return _text("请提供需要查询的订单号（例如 A102）。")
            calls = [("get_order_status", {"order_id": i}) for i in dict.fromkeys(ids)]
            if "dup" in faults:
                calls = calls + calls[:1]
            if "get_order_status" not in tool_names:
                return _text("当前没有可用的查询工具。")
            return _calls(calls)

        # 已拿到工具结果：根据结果组织最终回答
        lines = []
        for r in results:
            try:
                data = json.loads(r["content"])
            except json.JSONDecodeError:
                data = {"status": "error"}
            status = data.get("status")
            oid = (data.get("data") or {}).get("order_id") or data.get("order_id") or "?"
            if status == "ok":
                d = data["data"]
                lines.append(f"订单 {d['order_id']} 当前状态：{d['status']}（{d['detail']}）。")
            elif status == "permission_denied":
                lines.append(f"订单 {oid}：你没有权限查询该订单。")
            elif status == "not_found":
                lines.append(f"未找到订单 {oid}。")
            elif status in ("timeout", "unavailable"):
                lines.append("订单系统暂时不可用，请稍后再试。")
            elif status == "duplicate_call":
                continue
            else:
                lines.append(f"无法完成查询（{status}）。")
        return _text("\n".join(dict.fromkeys(lines)) or "没有可用的查询结果。")

    # ------------------------------------------------------------------ 响应封装

    def _json_response(self, body: dict, decided: dict[str, Any]) -> httpx.Response:
        prompt_tokens = sum(estimate_text(m.get("content") or "") + 4 for m in body["messages"])
        content = decided["message"].get("content") or ""
        args = "".join(c["function"]["arguments"] for c in decided["message"].get("tool_calls", []))
        rid = f"req_mock_{uuid.uuid4().hex[:12]}"
        payload = {
            "id": f"chatcmpl-mock-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": body["model"],
            "choices": [{"index": 0, **decided}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": estimate_text(content + args)},
        }
        return httpx.Response(200, json=payload, headers={"x-request-id": rid})

    def _stream_response(self, body: dict, faults: set[str]) -> httpx.Response:
        messages = body["messages"]
        text = self._chat_reply(messages, _latest(messages, "user"))
        delay = 0.3 if "slow" in faults else self.chunk_delay
        include_usage = (body.get("stream_options") or {}).get("include_usage")
        prompt_tokens = sum(estimate_text(m.get("content") or "") + 4 for m in messages)

        async def gen() -> AsyncIterator[bytes]:
            pieces = [text[i : i + 4] for i in range(0, len(text), 4)]
            for n, piece in enumerate(pieces):
                if "stream_error" in faults and n == 3:
                    # 连接早已返回 200，错误只能以流内事件的形式出现
                    yield _sse({"error": {"message": "upstream overloaded", "type": "server_error"}})
                    return
                chunk = {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                raw = _sse(chunk)
                # 故意把一个事件拆成两个网络分片，验证解析器不会把半行当成完整事件
                yield raw[:7]
                yield raw[7:]
                await asyncio.sleep(delay)
            yield _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            if include_usage:
                yield _sse({"choices": [], "usage": {"prompt_tokens": prompt_tokens,
                                                     "completion_tokens": estimate_text(text)}})
            yield b"data: [DONE]\n\n"

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "x-request-id": f"req_mock_{uuid.uuid4().hex[:12]}"},
            content=gen(),
        )


def _latest(messages: list[dict], role: str) -> str:
    for m in reversed(messages):
        if m["role"] == role:
            return m.get("content") or ""
    return ""


def _system(messages: list[dict]) -> str:
    return "\n".join(m.get("content") or "" for m in messages if m["role"] == "system")


def _text(content: str) -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}


def _calls(calls: list[tuple[str, dict]]) -> dict[str, Any]:
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"call_{uuid.uuid4().hex[:10]}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
                for name, args in calls
            ],
        },
        "finish_reason": "tool_calls",
    }


def _error(status: int, message: str, code: str, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        json={"error": {"message": message, "type": code, "code": code}},
        headers={"x-request-id": f"req_mock_{uuid.uuid4().hex[:12]}", **(headers or {})},
    )


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()
