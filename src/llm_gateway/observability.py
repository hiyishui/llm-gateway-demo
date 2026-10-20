"""可观测性：结构化日志、trace ID 与指标汇总。

对应文章「错误处理与可靠性 · 可观测性」与综合练习的“统一日志”和“完成标准”：
- 每次模型调用记录 trace ID、供应商 request ID、模型、耗时、尝试次数、Token、结束状态、错误类别；
- 日志脱敏：不记录 Prompt 原文、密钥或 Authorization 头，用户 ID 只记录哈希；
- 汇总 P50/P95 延迟、Token、估算成本、结构化解析成功率和工具成功率（GET /metrics）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import threading
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

logger = logging.getLogger("llm_gateway")


class JsonFormatter(logging.Formatter):
    """一行一个 JSON，便于日志系统检索与聚合。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "event": record.getMessage(),
            "trace_id": trace_id_var.get(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


class _StdoutHandler(logging.StreamHandler):
    """每次输出时取当前的 sys.stdout（而非创建时的对象），便于被测试框架或进程管理器重定向。"""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property  # type: ignore[override]
    def stream(self):  # noqa: ANN201
        return sys.stdout

    @stream.setter
    def stream(self, _value) -> None:  # noqa: ANN001
        pass


def setup_logging(level: str = "INFO") -> None:
    handler = _StdoutHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields})


def hash_user(user_id: str) -> str:
    """日志里只出现用户 ID 的短哈希，既能关联同一用户的请求，又不暴露原值。"""
    return hashlib.sha256(user_id.encode()).hexdigest()[:12]


@dataclass
class CallRecord:
    """一次模型调用的诊断记录（不含任何 Prompt 或回答原文）。"""

    operation: str
    model: str
    provider_request_id: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    status: str = "ok"  # ok / error / cancelled
    error_category: str | None = None
    prompt_ref: str | None = None
    tool_calls: list[str] = field(default_factory=list)  # 只记录工具名


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return round(ordered[k], 1)


class Metrics:
    """进程内指标。生产环境应换成 Prometheus/OpenTelemetry 等系统，这里只为演示口径。"""

    def __init__(self, price_input_per_1m: float, price_output_per_1m: float) -> None:
        self._lock = threading.Lock()
        self.price_in = price_input_per_1m
        self.price_out = price_output_per_1m
        self.latencies: dict[str, list[float]] = defaultdict(list)
        self.calls: dict[str, int] = defaultdict(int)
        self.errors: dict[str, int] = defaultdict(int)
        self.input_tokens = 0
        self.output_tokens = 0
        self.extraction: dict[str, int] = defaultdict(int)  # ok / needs_review / failed
        self.tools: dict[str, int] = defaultdict(int)  # 各工具执行状态计数

    def record_call(self, rec: CallRecord) -> None:
        with self._lock:
            self.calls[rec.operation] += 1
            self.latencies[rec.operation].append(rec.latency_ms)
            self.input_tokens += rec.input_tokens
            self.output_tokens += rec.output_tokens
            if rec.status != "ok":
                self.errors[rec.error_category or rec.status] += 1
        log_event("model_call", **asdict(rec))

    def record_extraction(self, status: str) -> None:
        with self._lock:
            self.extraction[status] += 1

    def record_tool(self, status: str) -> None:
        with self._lock:
            self.tools[status] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            all_lat = [v for vals in self.latencies.values() for v in vals]
            ext_total = sum(self.extraction.values())
            tool_total = sum(self.tools.values())
            cost = self.input_tokens / 1e6 * self.price_in + self.output_tokens / 1e6 * self.price_out
            return {
                "model_calls": dict(self.calls),
                "latency_ms": {
                    "p50": _percentile(all_lat, 50),
                    "p95": _percentile(all_lat, 95),
                    "by_operation": {
                        op: {"p50": _percentile(v, 50), "p95": _percentile(v, 95)}
                        for op, v in self.latencies.items()
                    },
                },
                "tokens": {"input": self.input_tokens, "output": self.output_tokens},
                "estimated_cost_usd": round(cost, 6),
                "errors_by_category": dict(self.errors),
                "extraction": {
                    **self.extraction,
                    # “解析成功”= 通过格式与 Schema 两层校验（含需人工复核的结果）
                    "parse_success_rate": _rate(ext_total - self.extraction.get("failed", 0), ext_total),
                },
                "tools": {**self.tools, "success_rate": _rate(self.tools.get("ok", 0), tool_total)},
            }


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None

