"""HTTP 层：认证、错误映射、trace ID、日志脱敏、指标。"""

import pytest

from .conftest import ALICE, BOB, FAKE_KEY


def test_healthz_and_trace_id(client):
    resp = client.get("/healthz", headers={"x-trace-id": "abc123"})
    assert resp.status_code == 200
    assert resp.headers["x-trace-id"] == "abc123"


def test_requires_gateway_token(client):
    assert client.post("/chat", json={"message": "hi"}).status_code == 401
    assert client.post("/chat", json={"message": "hi"}, headers={"Authorization": "Bearer nope"}).status_code == 401


@pytest.mark.parametrize(
    ("message", "status", "category"),
    [
        ("   ", 400, "bad_input"),
        ("x" * 9000, 413, "input_too_long"),
        ("hi [fault:400]", 502, "invalid_request"),
        ("hi [fault:auth]", 502, "auth"),
        ("hi [fault:quota]", 503, "quota_exceeded"),
        ("hi [fault:429]", 503, "rate_limited"),
        ("hi [fault:500]", 502, "server_error"),
        ("hi [fault:timeout]", 504, "timeout"),
        ("hi [fault:network]", 502, "network"),
    ],
)
def test_error_paths_are_distinct(client, message, status, category):
    resp = client.post("/chat", json={"message": message}, headers=ALICE)
    assert resp.status_code == status
    body = resp.json()["error"]
    assert body["category"] == category
    assert body["trace_id"] == resp.headers["x-trace-id"]
    if category == "rate_limited":
        assert "retry-after" in resp.headers


def test_flaky_upstream_recovers_with_retry(client):
    resp = client.post("/chat", json={"message": "hi [fault:flaky]"}, headers=ALICE)
    assert resp.status_code == 200
    assert resp.json()["attempts"] == 2


def test_endpoints_happy_path(client):
    chat = client.post("/chat", json={"message": "你好"}, headers=ALICE).json()
    assert chat["status"] == "ok" and chat["prompt_ref"] == "chat@v1"
    extract = client.post("/extract", json={"ticket_text": "订单 A102 没收到"}, headers=ALICE).json()
    assert extract["data"]["order_id"] == "A102"
    tools = client.post("/tools", json={"message": "查 B205"}, headers=BOB).json()
    assert tools["trace"][0]["status"] == "ok"  # bob 查自己的订单
    with client.stream("POST", "/stream", json={"message": "你好"}, headers=ALICE) as s:
        assert s.headers["content-type"].startswith("text/event-stream")
        assert "event: done" in s.read().decode()


def test_stream_pre_start_error_uses_http_status(client):
    resp = client.post("/stream", json={"message": "hi [fault:429]"}, headers=ALICE)
    assert resp.status_code == 503


def test_metrics_report(client):
    client.post("/chat", json={"message": "你好"}, headers=ALICE)
    client.post("/extract", json={"ticket_text": "x [fault:invalid_json]"}, headers=ALICE)
    client.post("/tools", json={"message": "查 A102"}, headers=ALICE)
    m = client.get("/metrics", headers=ALICE).json()
    assert m["latency_ms"]["p95"] >= m["latency_ms"]["p50"] >= 0
    assert m["tokens"]["input"] > 0 and m["estimated_cost_usd"] > 0
    assert m["extraction"]["parse_success_rate"] == 0.0
    assert m["tools"]["success_rate"] == 1.0


def test_logs_never_contain_secrets_or_prompts(client, capsys):
    client.post("/chat", json={"message": "我的手机号是 13800000000"}, headers=ALICE)
    client.post("/chat", json={"message": "hi [fault:auth]"}, headers=ALICE)
    client.post("/tools", json={"message": "查 A102"}, headers=ALICE)
    out = capsys.readouterr().out
    assert '"event": "model_call"' in out  # 确实输出了日志
    assert FAKE_KEY not in out
    assert "t-alice" not in out and '"alice"' not in out  # 只出现用户哈希
    assert "13800000000" not in out  # 不记录 Prompt 原文
