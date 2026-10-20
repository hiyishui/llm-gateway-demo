"""示例业务系统：内存中的订单库，以及一个只读的订单查询工具。

对应文章「上下文管理 · 显式状态」：订单号、状态这类业务事实保存在可信系统里，
每次需要时由工具实时查询，而不是依赖模型“记得”。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .registry import Tool, ToolContext, ToolResult


@dataclass(frozen=True)
class Order:
    order_id: str
    owner: str  # 订单所属用户，用于授权
    status: str
    detail: str


class OrderStore:
    # 两个特殊订单用于故障演示：Z999 模拟后端很慢（触发工具超时），Z500 模拟后端不可用
    SLOW_ORDER = "Z999"
    BROKEN_ORDER = "Z500"

    def __init__(self, slow_seconds: float = 10.0) -> None:
        self.slow_seconds = slow_seconds
        self._orders = {
            o.order_id: o
            for o in [
                Order("A102", "alice", "已发货", "顺丰 SF1234567，预计明天送达"),
                Order("A103", "alice", "待发货", "仓库备货中，预计 48 小时内发出"),
                Order("A104", "alice", "退款处理中", "退款申请已受理，3～5 个工作日原路退回"),
                Order("B205", "bob", "已签收", "已于昨天签收"),
                Order(self.SLOW_ORDER, "alice", "未知", "-"),
                Order(self.BROKEN_ORDER, "alice", "未知", "-"),
            ]
        }

    def exists(self, order_id: str) -> bool:
        return order_id in self._orders

    async def get(self, order_id: str) -> Order | None:
        if order_id == self.SLOW_ORDER:
            await asyncio.sleep(self.slow_seconds)
        if order_id == self.BROKEN_ORDER:
            raise ConnectionError("order database unavailable")
        return self._orders.get(order_id)


def get_order_status_tool(store: OrderStore) -> Tool:
    async def handler(args: dict, ctx: ToolContext) -> ToolResult:
        order = await store.get(args["order_id"])
        if order is None:
            return ToolResult("not_found", {"order_id": args["order_id"]})
        # 授权使用调用者的真实身份（来自网关认证），而不是模型或用户在对话里的声明
        if order.owner != ctx.user_id:
            return ToolResult("permission_denied", {"order_id": args["order_id"]})
        return ToolResult(
            "ok", {"order_id": order.order_id, "status": order.status, "detail": order.detail}
        )

    return Tool(
        name="get_order_status",
        description="查询当前登录用户名下某个订单的状态与物流信息。只读。",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号，1 个大写字母加 3 位数字，如 A102",
                    "pattern": "^[A-Z][0-9]{3}$",
                }
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
        handler=handler,
        read_only=True,
    )
