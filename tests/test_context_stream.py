"""上下文管理与流式输出。"""

import pytest

from llm_gateway.errors import ErrorCategory, GatewayError
from llm_gateway.service import GatewayService
from llm_gateway.sse import SSEParser

from .conftest import fast_mock, make_settings


async def _never_disconnected() -> bool:
    return False


async def _collect(service: GatewayService, user: str, text: str, conv: str | None = None, disconnect=None):
    plan = await service.prepare_stream(user, conv, text)
    raw = b"".join([chunk async for chunk in service.stream(plan, disconnect or _never_disconnected)])
    return plan, [(e.event, e.data) for e in SSEParser().feed(raw.decode())]


# ---------------------------------------------------------------- 上下文


async def test_session_memory_and_user_isolation(service):
    first = await service.chat("alice", None, "你好，我叫小明")
    cid = first["conversation_id"]
    again = await service.chat("alice", cid, "我叫什么？")
    assert "小明" in again["reply"]
    # bob 拿到了 alice 的 conversation_id，也看不到她的历史
    other = await service.chat("bob", cid, "我叫什么？")
    assert "小明" not in other["reply"]
    assert other["context"]["history_messages"] == 0


async def test_long_history_is_summarized_and_facts_survive():
    svc = GatewayService(make_settings(context_max_tokens=1200, history_keep_recent=2), fast_mock().transport())
    cid = (await svc.chat("alice", None, "你好，我叫小红"))["conversation_id"]
    for i in range(8):
        await svc.chat("alice", cid, f"第 {i} 个问题：" + "退款政策是什么？" * 15)
    resp = await svc.chat("alice", cid, "我叫什么？")
    session = svc.sessions.get_or_create("alice", cid)
    assert session.summary is not None
    assert resp["context"]["estimated_input_tokens"] <= 1200
    assert "小红" in resp["reply"]  # 事实通过滚动摘要保留下来
    await svc.aclose()


async def test_input_larger_than_budget_is_rejected(service):
    with pytest.raises(GatewayError) as info:
        await service.chat("alice", None, "长" * 7900)
    assert info.value.category is ErrorCategory.INPUT_TOO_LONG


async def test_failed_turn_is_not_saved(service):
    cid = (await service.chat("alice", None, "你好"))["conversation_id"]
    resp = await service.chat("alice", cid, "回答我 [fault:truncate]")
    assert resp["status"] == "truncated"
    assert len(service.sessions.get_or_create("alice", cid).messages) == 2


async def test_delete_session(service):
    cid = (await service.chat("alice", None, "你好"))["conversation_id"]
    assert service.sessions.delete("bob", cid) is False  # 不能删别人的会话
    assert service.sessions.delete("alice", cid) is True


# ---------------------------------------------------------------- 流式


async def test_stream_happy_path_saves_turn(service):
    plan, events = await _collect(service, "alice", "讲讲退款流程")
    kinds = [k for k, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "done" and "delta" in kinds
    assert '"complete": true' in events[-1][1]
    assert len(plan.session.messages) == 2


async def test_stream_mid_error_reports_partial_and_does_not_save(service):
    plan, events = await _collect(service, "alice", "讲讲退款流程 [fault:stream_error]")
    kind, data = events[-1]
    assert kind == "error"
    assert '"partial": true' in data
    assert plan.session.messages == []


async def test_stream_errors_before_start_raise_normally(service):
    # 建流阶段的错误在返回 200 之前抛出，路由层可以返回正常的 HTTP 状态码
    with pytest.raises(GatewayError) as info:
        await service.prepare_stream("alice", None, "hi [fault:auth]")
    assert info.value.category is ErrorCategory.AUTH


async def test_stream_client_disconnect_stops_and_is_recorded(service):
    calls = {"n": 0}

    async def disconnect_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    plan, events = await _collect(service, "alice", "讲讲退款流程", disconnect=disconnect_after_two)
    assert [k for k, _ in events] == ["start", "delta"]
    assert plan.session.messages == []  # 未完成的回答不写入历史
    assert service.metrics.errors.get("cancelled") == 1


async def test_stream_consumer_abort_releases_upstream(service):
    plan = await service.prepare_stream("alice", None, "讲讲退款流程")
    gen = service.stream(plan, _never_disconnected)
    await anext(gen)
    await gen.aclose()  # 模拟连接被服务器框架取消
    assert service.metrics.errors.get("cancelled") == 1
