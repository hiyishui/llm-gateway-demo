"""粗略的 Token 估算。

对应文章「上下文管理 · Token 预算」：Token 与字符数不是一一对应，准确值以供应商返回的 usage
或其官方分词器为准。这里用一个保守的近似值做“预算闸门”：中日韩字符按 1 个 Token 计，
其余字符按约 4 个字符 1 个 Token 计，每条消息再加少量结构开销。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from .models import Message, ToolDefinition

_CJK = re.compile(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
MESSAGE_OVERHEAD = 4


def estimate_text(text: str | None) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def estimate_messages(messages: Iterable[Message]) -> int:
    total = 0
    for m in messages:
        total += MESSAGE_OVERHEAD + estimate_text(m.content)
        for call in m.tool_calls:
            total += estimate_text(call.name) + estimate_text(call.arguments_json)
    return total


def estimate_tools(tools: Iterable[ToolDefinition]) -> int:
    """工具定义也会占用上下文，需要计入预算。"""
    return sum(
        estimate_text(json.dumps({"n": t.name, "d": t.description, "p": t.parameters}, ensure_ascii=False))
        for t in tools
    )
