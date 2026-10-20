"""应用内部的统一请求/响应类型。

对应文章「模型 API · API Key 与服务边界」：
接入多个供应商时，先定义自己的统一类型，再由适配器翻译供应商格式。
统一层要保留工具调用、拒绝原因、结束原因、用量等语义，而不是只剩一段文本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """模型提出的一次工具调用请求（只是“请求”，是否执行由应用决定）。"""

    id: str
    name: str
    arguments_json: str  # 保留模型给出的原始参数字符串，解析与校验在工具层完成


@dataclass
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)  # 仅 assistant 消息
    tool_call_id: str | None = None  # 仅 tool 消息：结果要关联到对应的调用


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class StructuredOutputSpec:
    """要求模型按 JSON Schema 返回最终答案。"""

    name: str
    schema: dict[str, Any]


@dataclass
class ModelRequest:
    messages: list[Message]
    tools: list[ToolDefinition] = field(default_factory=list)
    structured_output: StructuredOutputSpec | None = None
    max_output_tokens: int | None = None
    # 请求所属的业务操作（chat/extract/tools/stream），用于日志与指标分组
    operation: str = "chat"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


# 统一的结束原因：stop 正常结束；length 被输出上限截断；tool_calls 需要应用执行工具；
# content_filter 被安全策略拦截；unknown 供应商返回了未识别的值
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "unknown"]


@dataclass
class ModelResponse:
    text: str | None
    tool_calls: list[ToolCall]
    finish_reason: FinishReason
    usage: Usage
    model: str
    provider_request_id: str | None = None
    refusal: str | None = None  # 模型明确拒绝时的说明（如供应商提供）
    latency_ms: float = 0.0
    attempts: int = 1

    def as_assistant_message(self) -> Message:
        """把模型这一轮的输出（含工具调用）放回消息历史，供下一轮继续。"""
        return Message(role="assistant", content=self.text, tool_calls=list(self.tool_calls))


# ---- 流式事件：适配器把供应商的 SSE 分片翻译成以下统一事件 ----


@dataclass
class StreamStart:
    model: str
    provider_request_id: str | None


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallDelta:
    """流式中的工具调用参数是分多段到达的，只有结束后拼起来才是完整 JSON。"""

    index: int
    id: str | None
    name: str | None
    arguments_fragment: str


@dataclass
class StreamEnd:
    finish_reason: FinishReason
    usage: Usage | None


StreamEvent = StreamStart | TextDelta | ToolCallDelta | StreamEnd
