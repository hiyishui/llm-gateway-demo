"""SSE（Server-Sent Events）解析与生成。

对应文章「流式输出」：网络分片不等于事件——一个分片可能只有半行，也可能包含多条事件。
解析器必须先缓冲，按空行切出完整事件后再交给上层。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class SSEEvent:
    event: str  # 未指定 event 字段时默认为 "message"
    data: str


class SSEParser:
    """增量 SSE 解析器：feed() 任意大小的文本分片，返回其中已完整的事件。"""

    def __init__(self) -> None:
        self._buffer = ""
        self._event = "message"
        self._data_lines: list[str] = []

    def feed(self, chunk: str) -> list[SSEEvent]:
        self._buffer += chunk
        events: list[SSEEvent] = []
        # 只处理到最后一个换行符为止；之后的半行留在缓冲区等待下一个分片
        while True:
            newline = self._find_newline()
            if newline is None:
                break
            line, self._buffer = self._buffer[: newline[0]], self._buffer[newline[1] :]
            event = self._handle_line(line)
            if event:
                events.append(event)
        return events

    def flush(self) -> list[SSEEvent]:
        """连接结束时调用：处理没有以空行结尾的最后一个事件。"""
        events: list[SSEEvent] = []
        if self._buffer:
            event = self._handle_line(self._buffer.rstrip("\r"))
            self._buffer = ""
            if event:
                events.append(event)
        if self._data_lines:
            events.append(self._dispatch())
        return events

    def _find_newline(self) -> tuple[int, int] | None:
        # SSE 允许 \n、\r\n、\r 三种换行；返回 (行结束位置, 下一行开始位置)
        for i, ch in enumerate(self._buffer):
            if ch == "\n":
                return i, i + 1
            if ch == "\r":
                if i + 1 == len(self._buffer):
                    return None  # \r 可能是 \r\n 的前半，等待更多数据
                return i, i + 2 if self._buffer[i + 1] == "\n" else i + 1
        return None

    def _handle_line(self, line: str) -> SSEEvent | None:
        if line == "":
            return self._dispatch() if self._data_lines else None
        if line.startswith(":"):
            return None  # 注释行，常用作心跳
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "data":
            self._data_lines.append(value)
        elif name == "event":
            self._event = value
        return None

    def _dispatch(self) -> SSEEvent:
        event = SSEEvent(self._event, "\n".join(self._data_lines))
        self._event, self._data_lines = "message", []
        return event


async def iter_sse(byte_chunks: AsyncIterator[bytes]) -> AsyncIterator[SSEEvent]:
    """把字节流转换为 SSE 事件流。使用增量解码，避免把多字节中文字符从中间切断。"""
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")()
    parser = SSEParser()
    async for chunk in byte_chunks:
        for event in parser.feed(decoder.decode(chunk)):
            yield event
    for event in parser.feed(decoder.decode(b"", final=True)) + parser.flush():
        yield event


def format_sse(event: str, data: dict[str, Any]) -> bytes:
    """生成发给浏览器的 SSE 事件。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode()
