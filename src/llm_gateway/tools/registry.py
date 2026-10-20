"""工具注册表。

对应文章「Tool Calling · 工具安全规则」，execute() 依次做：
1. allowlist：只执行本轮明确允许、且已在服务端注册的工具，绝不按模型给的字符串动态找函数；
2. 参数校验：解析 JSON 并按 JSON Schema 校验，拒绝未知字段、越界值和非法格式；
3. 副作用控制：非只读工具不自动执行，返回 requires_confirmation 交给人工确认；
4. 重复调用：同一轮任务中名称和参数完全相同的调用不重复执行；
5. 超时与异常：单次执行有超时，异常转换成明确状态，不伪装成成功；
6. 结果限长并记录审计日志。
授权（用户能否访问某个资源）在各工具的 handler 中用真实身份完成。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from ..models import ToolCall, ToolDefinition
from ..observability import Metrics, hash_user, log_event


@dataclass
class ToolContext:
    user_id: str  # 来自网关认证的真实用户身份
    # 本轮任务内已执行过的调用：(工具名, 规范化参数) → 结果，用于识别重复调用
    executed: dict[tuple[str, str], ToolResult] = field(default_factory=dict)


@dataclass
class ToolResult:
    # ok / not_found / permission_denied / invalid_arguments / unknown_tool /
    # requires_confirmation / duplicate_call / timeout / unavailable
    status: str
    data: dict[str, Any] | None = None
    message: str | None = None

    def to_model_content(self, max_chars: int) -> str:
        """序列化后交回模型。结果过长时截断，并明确标注，避免模型误以为数据完整。"""
        text = json.dumps(
            {"status": self.status, "data": self.data, "message": self.message}, ensure_ascii=False
        )
        if len(text) > max_chars:
            text = json.dumps({"status": self.status, "truncated": True, "preview": text[:max_chars]},
                              ensure_ascii=False)
        return text


Handler = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    read_only: bool = True

    def __post_init__(self) -> None:
        Draft202012Validator.check_schema(self.parameters)
        self._validator = Draft202012Validator(self.parameters)

    def definition(self) -> ToolDefinition:
        return ToolDefinition(self.name, self.description, self.parameters)


class ToolRegistry:
    def __init__(self, metrics: Metrics, timeout_seconds: float, result_max_chars: int) -> None:
        self._tools: dict[str, Tool] = {}
        self.metrics = metrics
        self.timeout_seconds = timeout_seconds
        self.result_max_chars = result_max_chars

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def definitions(self, allowed: set[str]) -> list[ToolDefinition]:
        """只把本轮允许的工具告诉模型：模型看不到的工具，就不会被提议调用。"""
        return [t.definition() for name, t in self._tools.items() if name in allowed]

    async def execute(self, call: ToolCall, allowed: set[str], ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        result = await self._execute(call, allowed, ctx)
        self.metrics.record_tool(result.status)
        log_event(
            "tool_call",
            level=logging.INFO if result.status == "ok" else logging.WARNING,
            tool=call.name,
            status=result.status,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            user=hash_user(ctx.user_id),
        )
        return result

    async def _execute(self, call: ToolCall, allowed: set[str], ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None or call.name not in allowed:
            return ToolResult("unknown_tool", message=f"tool '{call.name}' is not available")

        try:
            args = json.loads(call.arguments_json or "{}")
        except json.JSONDecodeError:
            return ToolResult("invalid_arguments", message="arguments are not valid JSON")
        errors = sorted(tool._validator.iter_errors(args), key=lambda e: list(e.path))
        if errors:
            return ToolResult("invalid_arguments", message="; ".join(e.message for e in errors[:3]))

        if not tool.read_only:
            # 写操作（支付、删除、发信……）需要人工确认 + 幂等键，本示例一律不自动执行
            return ToolResult("requires_confirmation", message="side-effect tool needs human approval")

        key = (call.name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if key in ctx.executed:
            return ToolResult("duplicate_call", message="identical call already executed in this task")

        try:
            result = await asyncio.wait_for(tool.handler(args, ctx), timeout=self.timeout_seconds)
        except TimeoutError:
            result = ToolResult("timeout", message="tool execution timed out")
        except Exception as exc:  # noqa: BLE001 —— 工具内部任何异常都要转成明确状态
            result = ToolResult("unavailable", message=type(exc).__name__)
        ctx.executed[key] = result
        return result
