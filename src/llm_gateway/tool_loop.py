"""Tool Calling 调用循环。

对应文章「Tool Calling · 标准调用循环」：
    应用发送输入与工具定义 → 模型提出工具调用 → 应用校验并执行 → 结果交回模型 → ……
    直到模型给出最终回答，或触达轮数 / 总耗时上限。
模型只“提议”，真正执行与否、怎么执行，全部由 ToolRegistry 决定。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .gateway import ModelGateway
from .models import Message, ModelRequest
from .tools.registry import ToolContext, ToolRegistry


@dataclass
class ToolTraceItem:
    round: int
    tool: str
    status: str
    duration_ms: float


@dataclass
class ToolLoopResult:
    # completed：模型给出了最终回答；refused / truncated：模型输出不可用；
    # loop_limit / time_limit：触达预算后停止
    status: Literal["completed", "refused", "truncated", "loop_limit", "time_limit"]
    answer: str
    rounds: int
    trace: list[ToolTraceItem] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_tool_loop(
    gateway: ModelGateway,
    registry: ToolRegistry,
    messages: list[Message],
    allowed_tools: set[str],
    ctx: ToolContext,
    *,
    max_rounds: int,
    total_timeout: float,
    prompt_ref: str | None = None,
) -> ToolLoopResult:
    deadline = time.monotonic() + total_timeout
    definitions = registry.definitions(allowed_tools)
    trace: list[ToolTraceItem] = []
    usage = {"input_tokens": 0, "output_tokens": 0}

    for round_no in range(1, max_rounds + 1):
        resp = await gateway.generate(
            ModelRequest(messages, tools=definitions, operation="tools"), prompt_ref=prompt_ref
        )
        usage["input_tokens"] += resp.usage.input_tokens
        usage["output_tokens"] += resp.usage.output_tokens

        if resp.refusal:
            return ToolLoopResult("refused", resp.refusal, round_no, trace, usage)
        if not resp.tool_calls:
            if resp.finish_reason == "length":
                return ToolLoopResult("truncated", resp.text or "", round_no, trace, usage)
            return ToolLoopResult("completed", resp.text or "", round_no, trace, usage)

        # 先把模型的“调用请求”放回历史，再逐个追加结果，并用 tool_call_id 一一对应
        messages.append(resp.as_assistant_message())
        for call in resp.tool_calls:
            started = time.perf_counter()
            result = await registry.execute(call, allowed_tools, ctx)
            trace.append(ToolTraceItem(round_no, call.name, result.status,
                                       round((time.perf_counter() - started) * 1000, 1)))
            messages.append(
                Message("tool", result.to_model_content(registry.result_max_chars), tool_call_id=call.id)
            )

        if time.monotonic() > deadline:
            return ToolLoopResult("time_limit", "处理时间超过上限，已停止本次查询，请稍后重试。",
                                  round_no, trace, usage)

    return ToolLoopResult("loop_limit", "工具调用次数超过上限，已停止本次查询。", max_rounds, trace, usage)
