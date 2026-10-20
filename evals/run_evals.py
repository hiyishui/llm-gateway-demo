"""固定案例评测：对网关的四个接口运行 evals/cases.jsonl，并输出质量与性能报告。

对应文章「Prompt 设计 · 用实验而不是直觉调 Prompt」与综合练习的“完成标准”：
- 案例覆盖正常、缺字段、长输入、拒绝、无答案、恶意指令和故障注入；
- 每次运行记录模型、Prompt 版本、每条案例的结果，便于对比改动前后；
- 汇总 P50/P95 延迟、Token、估算成本、结构化解析成功率和工具成功率。

用法：
    uv run python evals/run_evals.py            # 使用 .env 中的配置（默认 mock）
    uv run python evals/run_evals.py --only extract
带 mock_only 标记的案例依赖 Mock 的故障注入标记，接入真实模型时自动跳过。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from llm_gateway.api import create_app
from llm_gateway.config import load_settings
from llm_gateway.sse import SSEParser

ROOT = Path(__file__).parent
DEMO_USERS = {"alice": "eval-token-alice", "bob": "eval-token-bob"}


def build_input(case: dict[str, Any]) -> str:
    if "input_repeat" in case:
        text, times = case["input_repeat"]
        return text * times
    return case.get("input", "")


def call(client: TestClient, case: dict[str, Any]) -> tuple[int, Any]:
    """按案例调用对应接口，返回 (HTTP 状态码, 解析后的响应)。"""
    user = case.get("user", "alice")
    headers = {"Authorization": f"Bearer {DEMO_USERS[user]}"}
    endpoint = case["endpoint"]

    if endpoint == "chat" and "turns" in case:
        conv_id, resp = None, None
        for turn in case["turns"]:
            turn = {"user": user, "message": turn} if isinstance(turn, str) else turn
            h = {"Authorization": f"Bearer {DEMO_USERS[turn['user']]}"}
            resp = client.post("/chat", json={"message": turn["message"], "conversation_id": conv_id}, headers=h)
            conv_id = resp.json().get("conversation_id", conv_id)
        return resp.status_code, resp.json()

    text = build_input(case)
    if endpoint == "chat":
        r = client.post("/chat", json={"message": text}, headers=headers)
    elif endpoint == "extract":
        r = client.post("/extract", json={"ticket_text": text}, headers=headers)
    elif endpoint == "tools":
        r = client.post("/tools", json={"message": text}, headers=headers)
    elif endpoint == "stream":
        with client.stream("POST", "/stream", json={"message": text}, headers=headers) as s:
            body = s.read().decode()
            if s.status_code != 200:
                return s.status_code, json.loads(body)
            events = [(e.event, json.loads(e.data)) for e in SSEParser().feed(body)]
            return s.status_code, {"events": events}
    else:
        raise ValueError(endpoint)
    return r.status_code, r.json()


def check(case: dict[str, Any], status: int, body: Any) -> list[str]:
    """逐项比对期望，返回不满足的项（空列表表示通过）。"""
    exp, fails = case["expect"], []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    expect(status == exp["http"], f"http {status} != {exp['http']}")
    if status != exp["http"]:
        return fails
    if "error" in exp:
        expect(body.get("error", {}).get("category") == exp["error"], f"error {body.get('error')}")
        return fails

    data = body.get("data") or {}
    text = body.get("reply") or body.get("answer") or ""
    trace = [t["status"] for t in body.get("trace", [])]
    events = body.get("events", [])
    for key, want in exp.items():
        if key == "status":
            expect(body.get("status") == want, f"status {body.get('status')} != {want}")
        elif key == "status_in":
            expect(body.get("status") in want, f"status {body.get('status')} not in {want}")
        elif key in ("category", "order_id"):
            expect(data.get(key) == want, f"{key} {data.get(key)!r} != {want!r}")
        elif key == "failed_layer":
            expect(body.get("failed_layer") == want, f"failed_layer {body.get('failed_layer')}")
        elif key == "issues_contain":
            expect(set(want) <= set(body.get("business_issues", [])), f"issues {body.get('business_issues')}")
        elif key == "no_extra_fields":
            expect(set(data) <= {"category", "order_id", "summary", "needs_human_review"}, f"fields {set(data)}")
        elif key in ("reply_contains", "answer_contains"):
            expect(all(w in text for w in want), f"text missing {want}")
        elif key in ("reply_not_contains", "answer_not_contains"):
            expect(not any(w in text for w in want), f"text contains forbidden {want}")
        elif key == "trace":
            expect(trace == want, f"trace {trace} != {want}")
        elif key == "trace_tools_subset":
            tools = {t["tool"] for t in body.get("trace", [])}
            expect(tools <= set(want), f"tools {tools} not within {want}")
        elif key == "attempts_at_least":
            expect(body.get("attempts", 0) >= want, f"attempts {body.get('attempts')}")
        elif key == "events_start":
            expect(bool(events) and events[0][0] == want, f"first event {events[:1]}")
        elif key == "events_end":
            expect(bool(events) and events[-1][0] == want, f"last event {events[-1:]}")
        elif key == "complete":
            expect(bool(events) and events[-1][1].get("complete") is want, "stream not complete")
        elif key == "partial":
            expect(bool(events) and events[-1][1].get("partial") is want, "partial flag mismatch")
    return fails


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="只运行某个接口的案例：chat/extract/tools/stream")
    args = parser.parse_args()

    settings = load_settings()
    # 评测使用独立的网关令牌，不依赖 .env 中的 GATEWAY_TOKENS；关闭日志以保持输出整洁
    settings = replace(settings, gateway_tokens={v: k for k, v in DEMO_USERS.items()}, log_level="CRITICAL")
    is_mock = settings.llm_provider == "mock"
    if is_mock:
        # Mock 模式下缩短退避与工具超时，只验证处理路径，不必真实等待
        settings = replace(settings, llm_retry_base_delay=0.02, llm_retry_max_delay=0.1, tool_timeout_seconds=1.0)

    cases = [json.loads(line) for line in (ROOT / "cases.jsonl").read_text("utf-8").splitlines() if line.strip()]
    if args.only:
        cases = [c for c in cases if c["endpoint"] == args.only]

    results, skipped = [], 0
    app = create_app(settings)
    prompt_refs = app.state.service.prompts.latest_refs()
    with TestClient(app) as client:
        for case in cases:
            if case.get("mock_only") and not is_mock:
                skipped += 1
                continue
            started = time.perf_counter()
            status, body = call(client, case)
            fails = check(case, status, body)
            results.append({
                "id": case["id"], "group": case["group"], "endpoint": case["endpoint"],
                "passed": not fails, "failures": fails,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            })
            mark = "PASS" if not fails else "FAIL"
            print(f"[{mark}] {case['id']:<34} {'; '.join(fails)}")
        metrics = client.get("/metrics", headers={"Authorization": f"Bearer {DEMO_USERS['alice']}"}).json()

    passed = sum(r["passed"] for r in results)
    by_group: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        by_group[r["group"]]["passed" if r["passed"] else "failed"] += 1

    print("\n==== 汇总 ====")
    print(f"Prompt 版本: {', '.join(prompt_refs)}")
    print(f"模型: {settings.llm_provider}/{settings.llm_model}  案例: {len(results)}  通过: {passed}  跳过: {skipped}")
    for group, cnt in sorted(by_group.items()):
        print(f"  {group:<6} 通过 {cnt['passed']}/{cnt['passed'] + cnt['failed']}")
    lat = metrics["latency_ms"]
    print(f"模型调用延迟 P50/P95: {lat['p50']} / {lat['p95']} ms")
    print(f"Token 输入/输出: {metrics['tokens']['input']} / {metrics['tokens']['output']}  "
          f"估算成本: ${metrics['estimated_cost_usd']}")
    print(f"结构化解析成功率: {metrics['extraction'].get('parse_success_rate')}  "
          f"工具成功率: {metrics['tools'].get('success_rate')}")
    print(f"错误分类: {metrics['errors_by_category']}")

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report = report_dir / f"report-{time.strftime('%Y%m%d-%H%M%S')}.json"
    report.write_text(json.dumps({
        "provider": settings.llm_provider, "model": settings.llm_model,
        "prompts": prompt_refs,
        "passed": passed, "total": len(results), "skipped": skipped,
        "results": results, "metrics": metrics,
    }, ensure_ascii=False, indent=2), "utf-8")
    print(f"报告已写入 {report.relative_to(ROOT.parent)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
