"""适配器接口：业务代码只依赖这个接口，不直接接触任何供应商的字段名。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from ..models import ModelRequest, ModelResponse, StreamEvent


class ModelAdapter(Protocol):
    model: str

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """非流式调用：等完整结果生成后返回。失败时抛出 GatewayError。"""
        ...

    def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """流式调用：逐个产出统一的流式事件。流内错误同样以 GatewayError 抛出。"""
        ...

    async def aclose(self) -> None: ...
