"""配置加载：所有可变参数都来自环境变量（.env）。

对应文章「模型 API · 关键工程概念」：
- 模型标识不写死在业务逻辑里，而是放在配置层，方便切换与路由；
- API Key 只由后端持有，并且用 Secret 包装，避免被 print / 日志意外输出。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


class Secret:
    """包装敏感字符串：repr/str 永远显示为掩码，只有显式调用 reveal() 才拿到明文。"""

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return "Secret('***')" if self._value else "Secret('')"

    __str__ = __repr__


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _parse_tokens(raw: str) -> dict[str, str]:
    """解析 GATEWAY_TOKENS="token1:alice,token2:bob" 为 {token: user_id}。"""
    mapping: dict[str, str] = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        token, _, user = pair.partition(":")
        if token and user:
            mapping[token] = user
    return mapping


@dataclass(frozen=True)
class Settings:
    # ---- 模型后端 ----
    llm_provider: str = "mock"  # mock：离线模拟服务；openai：任意 OpenAI 兼容服务
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: Secret = field(default_factory=lambda: Secret(""))
    llm_model: str = "gpt-4o-mini"
    # 不同供应商对输出上限字段的命名不同（max_tokens / max_completion_tokens）
    llm_max_tokens_param: str = "max_tokens"
    llm_max_output_tokens: int = 800
    # 结构化输出模式：json_schema（服务端约束）/ json_object / prompt（仅靠提示词）
    llm_structured_output_mode: str = "json_schema"
    # 流式时是否请求在最后一个分片返回 usage（部分兼容服务不支持）
    llm_stream_include_usage: bool = True

    # ---- 超时与重试 ----
    llm_connect_timeout: float = 5.0
    llm_read_timeout: float = 60.0
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 0.5
    llm_retry_max_delay: float = 8.0

    # ---- 上下文与预算 ----
    context_max_tokens: int = 8000  # 模型上下文上限（按所选模型填写）
    history_keep_recent: int = 6  # 滑动窗口：至少保留的最近消息数
    session_ttl_seconds: int = 1800
    max_input_chars: int = 8000  # 单条用户输入的硬上限，超过直接拒绝

    # ---- Tool Calling 预算 ----
    tool_max_rounds: int = 4
    tool_timeout_seconds: float = 5.0
    tool_total_timeout_seconds: float = 30.0
    tool_result_max_chars: int = 2000

    # ---- 网关自身的认证与计费估算 ----
    gateway_tokens: dict[str, str] = field(default_factory=dict)
    price_input_per_1m: float = 0.15
    price_output_per_1m: float = 0.60

    log_level: str = "INFO"
    prompt_dir: Path = Path(__file__).parent / "prompts"


def load_settings(env_file: str | os.PathLike[str] | None = ".env") -> Settings:
    """从 .env + 进程环境变量构造 Settings。进程环境变量优先（便于部署时覆盖）。"""
    if env_file:
        load_dotenv(env_file, override=False)
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "mock").strip().lower(),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        llm_api_key=Secret(os.getenv("LLM_API_KEY", "")),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        llm_max_tokens_param=os.getenv("LLM_MAX_TOKENS_PARAM", "max_tokens"),
        llm_max_output_tokens=_int("LLM_MAX_OUTPUT_TOKENS", 800),
        llm_structured_output_mode=os.getenv("LLM_STRUCTURED_OUTPUT_MODE", "json_schema"),
        llm_stream_include_usage=_bool("LLM_STREAM_INCLUDE_USAGE", True),
        llm_connect_timeout=_float("LLM_CONNECT_TIMEOUT", 5.0),
        llm_read_timeout=_float("LLM_READ_TIMEOUT", 60.0),
        llm_max_retries=_int("LLM_MAX_RETRIES", 3),
        llm_retry_base_delay=_float("LLM_RETRY_BASE_DELAY", 0.5),
        llm_retry_max_delay=_float("LLM_RETRY_MAX_DELAY", 8.0),
        context_max_tokens=_int("CONTEXT_MAX_TOKENS", 8000),
        history_keep_recent=_int("HISTORY_KEEP_RECENT", 6),
        session_ttl_seconds=_int("SESSION_TTL_SECONDS", 1800),
        max_input_chars=_int("MAX_INPUT_CHARS", 8000),
        tool_max_rounds=_int("TOOL_MAX_ROUNDS", 4),
        tool_timeout_seconds=_float("TOOL_TIMEOUT_SECONDS", 5.0),
        tool_total_timeout_seconds=_float("TOOL_TOTAL_TIMEOUT_SECONDS", 30.0),
        tool_result_max_chars=_int("TOOL_RESULT_MAX_CHARS", 2000),
        gateway_tokens=_parse_tokens(os.getenv("GATEWAY_TOKENS", "")),
        price_input_per_1m=_float("PRICE_INPUT_PER_1M", 0.15),
        price_output_per_1m=_float("PRICE_OUTPUT_PER_1M", 0.60),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
