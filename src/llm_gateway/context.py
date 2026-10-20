"""上下文管理：会话状态、Token 预算、滑动窗口与滚动摘要。

对应文章「上下文管理」：
- 会话键 = (用户 ID, conversation_id)：不同用户即使拿到同一个 conversation_id 也看不到对方的历史；
- 会话有过期时间，也可以主动删除；
- 预算：固定指令 + 工具定义 + 历史 + 本轮输入 + 预留输出 ≤ 模型上下文上限；
- 超出预算时：把较早历史交给模型压缩成摘要（滚动摘要），只保留最近若干条（滑动窗口）；
  摘要失败时退化为直接丢弃最早的消息，保证请求仍然可以继续。
- 只有一轮完整成功后，才把本轮问答写回历史，失败或被截断的半截回答不会污染上下文。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from .config import Settings
from .errors import ErrorCategory, GatewayError
from .gateway import ModelGateway
from .models import Message, ModelRequest, ToolDefinition
from .observability import log_event
from .prompting import PromptRegistry
from .tokens import estimate_messages, estimate_text, estimate_tools

SUMMARY_PREFIX = "历史摘要（由模型根据较早对话生成，仅供参考；订单等业务事实以系统查询结果为准）：\n"


@dataclass
class Session:
    user_id: str
    conversation_id: str
    messages: list[Message] = field(default_factory=list)  # 只保存 user/assistant 的最终消息
    summary: str | None = None
    updated_at: float = field(default_factory=time.monotonic)


class SessionStore:
    """内存会话存储。生产环境可换成 Redis/数据库，但“按用户隔离 + 过期 + 可删除”的语义不变。"""

    def __init__(self, ttl_seconds: int) -> None:
        self.ttl = ttl_seconds
        self._sessions: dict[tuple[str, str], Session] = {}

    def get_or_create(self, user_id: str, conversation_id: str | None) -> Session:
        self._purge_expired()
        conversation_id = conversation_id or uuid.uuid4().hex
        key = (user_id, conversation_id)
        session = self._sessions.get(key)
        if session is None:
            session = self._sessions[key] = Session(user_id, conversation_id)
        return session

    def save_turn(self, session: Session, user_text: str, assistant_text: str) -> None:
        session.messages += [Message("user", user_text), Message("assistant", assistant_text)]
        session.updated_at = time.monotonic()

    def delete(self, user_id: str, conversation_id: str) -> bool:
        return self._sessions.pop((user_id, conversation_id), None) is not None

    def _purge_expired(self) -> None:
        now = time.monotonic()
        for key in [k for k, s in self._sessions.items() if now - s.updated_at > self.ttl]:
            del self._sessions[key]


@dataclass
class BuiltContext:
    messages: list[Message]
    estimated_input_tokens: int
    history_messages: int
    summarized_messages: int  # 本轮被压缩进摘要的消息数
    dropped_messages: int  # 本轮因预算不足被直接丢弃的消息数
    has_summary: bool


class ContextBuilder:
    def __init__(self, settings: Settings, gateway: ModelGateway, prompts: PromptRegistry) -> None:
        self.settings = settings
        self.gateway = gateway
        self.prompts = prompts

    async def build(
        self,
        session: Session,
        system: list[Message],
        user_text: str,
        tools: list[ToolDefinition] | None = None,
        reserved_output: int | None = None,
    ) -> BuiltContext:
        reserved = reserved_output or self.settings.llm_max_output_tokens
        fixed = estimate_messages(system) + estimate_tools(tools or [])
        new_input = estimate_messages([Message("user", user_text)])
        history_budget = self.settings.context_max_tokens - reserved - fixed - new_input
        if history_budget < 0:
            # 仅本轮输入就放不下：直接拒绝，并提示调用方先裁剪、分块或改用检索
            raise GatewayError(ErrorCategory.INPUT_TOO_LONG, "input exceeds the model context budget")

        summarized = dropped = 0
        if self._history_tokens(session) > history_budget:
            summarized, dropped = await self._shrink(session, history_budget)

        history = list(session.messages)
        if session.summary:
            history = [Message("system", SUMMARY_PREFIX + session.summary)] + history
        messages = system + history + [Message("user", user_text)]
        return BuiltContext(
            messages=messages,
            estimated_input_tokens=estimate_messages(messages) + estimate_tools(tools or []),
            history_messages=len(session.messages),
            summarized_messages=summarized,
            dropped_messages=dropped,
            has_summary=session.summary is not None,
        )

    @staticmethod
    def _history_tokens(session: Session) -> int:
        summary = estimate_text(SUMMARY_PREFIX + session.summary) if session.summary else 0
        return estimate_messages(session.messages) + summary

    async def _shrink(self, session: Session, budget: int) -> tuple[int, int]:
        keep = self.settings.history_keep_recent
        older, recent = session.messages[:-keep], session.messages[-keep:]
        summarized = 0
        if older:
            try:
                session.summary = await self._summarize(session.summary, older)
                summarized = len(older)
            except GatewayError as err:
                # 摘要本身失败（限流、超时……）不应拖垮整轮对话：退化为滑动窗口
                log_event("summary_failed", level=logging.WARNING, error_category=err.category.value)
            session.messages = recent

        dropped = len(older) - summarized
        # 摘要后仍超预算：继续按“成对”丢弃最早的问答（滑动窗口），直到放得下
        while session.messages and self._history_tokens(session) > budget:
            session.messages = session.messages[2:]
            dropped += 2
        if session.summary and self._history_tokens(session) > budget:
            session.summary = None  # 极端情况下连摘要也放不下
        return summarized, dropped

    async def _summarize(self, previous: str | None, messages: list[Message]) -> str:
        lines = [f"(已有摘要) {previous}"] if previous else []
        lines += [f"{m.role}: {m.content}" for m in messages if m.content]
        template = self.prompts.get("summarize")
        resp = await self.gateway.generate(
            ModelRequest(template.render(transcript="\n".join(lines)), max_output_tokens=300, operation="summarize"),
            prompt_ref=template.ref,
        )
        if not resp.text or resp.finish_reason != "stop":
            raise GatewayError(ErrorCategory.MODEL_OUTPUT_UNUSABLE, "summary unavailable")
        return resp.text.strip()
