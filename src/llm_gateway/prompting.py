"""Prompt 模板管理。

对应文章「Prompt 设计」：
- 固定指令与动态数据分开：模板存放在 prompts/<name>@<version>.txt，像代码一样做版本控制；
- 每次调用都记录使用的模板版本，方便在同一组案例上对比新旧版本；
- 动态数据放进明确的标签（如 <ticket>），并中和数据里伪造的闭合标签，
  防止用户文本“跳出”数据区。但这只是降低风险，真正的防线仍是应用层的校验与权限控制。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .models import Message

_VAR = re.compile(r"\{\{(\w+)\}\}")
_SECTION = re.compile(r"^\[(system|user)\]\s*$", re.M)
# 中和数据中的 <ticket>、</transcript> 等标签，避免伪造数据边界
_DATA_TAG = re.compile(r"</?\s*(ticket|transcript)\s*>", re.I)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system: str
    user: str | None

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def render(self, **variables: str) -> list[Message]:
        """渲染为消息列表。缺少变量直接报错，避免把 {{ticket_text}} 原样发给模型。"""

        def fill(text: str) -> str:
            def sub(m: re.Match[str]) -> str:
                key = m.group(1)
                if key not in variables:
                    raise KeyError(f"prompt {self.ref} missing variable: {key}")
                return _DATA_TAG.sub(lambda t: t.group(0).replace("<", "[").replace(">", "]"), variables[key])

            return _VAR.sub(sub, text)

        messages = [Message(role="system", content=fill(self.system))]
        if self.user is not None:
            messages.append(Message(role="user", content=fill(self.user)))
        return messages


class PromptRegistry:
    """从目录加载所有模板；get(name) 默认取最新版本，也可指定版本做 A/B 对比。"""

    def __init__(self, directory: Path) -> None:
        self._templates: dict[str, dict[str, PromptTemplate]] = {}
        for path in sorted(directory.glob("*@v*.txt")):
            name, version = path.stem.split("@")
            self._templates.setdefault(name, {})[version] = _parse(name, version, path.read_text("utf-8"))

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        versions = self._templates[name]
        if version is None:
            version = max(versions, key=lambda v: int(v.lstrip("v")))
        return versions[version]

    def latest_refs(self) -> list[str]:
        """当前生效（最新版本）的全部模板，评测报告会记录它们以便对比。"""
        return [self.get(name).ref for name in sorted(self._templates)]


def _parse(name: str, version: str, text: str) -> PromptTemplate:
    parts = _SECTION.split(text)
    # split 结果形如 ["", "system", "...", "user", "..."]
    sections = {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts) - 1, 2)}
    return PromptTemplate(name, version, sections["system"], sections.get("user"))
