"""SSE 解析：网络分片 ≠ 事件。"""

from llm_gateway.sse import SSEParser, iter_sse


def test_event_split_across_chunks():
    p = SSEParser()
    assert p.feed("data: {\"a\"") == []  # 半行：不能当成事件
    assert p.feed(": 1}\n") == []  # 行完整但事件尚未以空行结束
    events = p.feed("\n")
    assert [e.data for e in events] == ['{"a": 1}']


def test_multiple_events_in_one_chunk_and_crlf():
    p = SSEParser()
    events = p.feed("event: delta\r\ndata: x\r\n\r\n: heartbeat\n\ndata: y\n\n")
    assert [(e.event, e.data) for e in events] == [("delta", "x"), ("message", "y")]


def test_multiline_data_and_flush():
    p = SSEParser()
    assert p.feed("data: line1\ndata: line2") == []
    assert [e.data for e in p.flush()] == ["line1\nline2"]


async def test_iter_sse_does_not_break_multibyte_chars():
    raw = "data: 你好世界\n\n".encode()

    async def chunks():
        # 每次只送 1 个字节，会把中文字符的 UTF-8 编码从中间切开
        for i in range(len(raw)):
            yield raw[i : i + 1]

    events = [e async for e in iter_sse(chunks())]
    assert [e.data for e in events] == ["你好世界"]
