"""结构化输出：工单字段抽取与三层校验。

对应文章「结构化输出」：
1. 格式有效：响应能否解析为 JSON（还要先排除拒绝、截断、空输出）；
2. Schema 有效：字段、类型、必填、枚举是否满足 Schema（即使服务端声称 strict，本地仍要再校验）；
3. 业务有效：订单是否存在、订单号是否真的出现在原文里、分类与字段是否一致——只能由应用代码检查。
格式正确 ≠ 事实正确：第三层发现问题时转人工复核（needs_review），而不是直接当作成功。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from jsonschema import Draft202012Validator

from .gateway import ModelGateway
from .models import ModelRequest, StructuredOutputSpec
from .observability import Metrics
from .prompting import PromptRegistry
from .tools.orders import OrderStore

TICKET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["退款", "物流", "产品问题", "其他"]},
        "order_id": {"type": ["string", "null"]},
        "summary": {"type": "string"},
        "needs_human_review": {"type": "boolean"},
    },
    "required": ["category", "order_id", "summary", "needs_human_review"],
    "additionalProperties": False,
}
_validator = Draft202012Validator(TICKET_SCHEMA)
ORDER_ID_FORMAT = re.compile(r"^[A-Z][0-9]{3}$")

Status = Literal["ok", "needs_review", "failed"]


@dataclass
class ExtractionResult:
    status: Status
    data: dict[str, Any] | None
    failed_layer: Literal["format", "schema"] | None = None
    errors: list[str] = field(default_factory=list)  # 格式/Schema 层错误
    business_issues: list[str] = field(default_factory=list)  # 业务层发现的问题
    finish_reason: str | None = None
    prompt_ref: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TicketExtractor:
    def __init__(self, gateway: ModelGateway, prompts: PromptRegistry, orders: OrderStore, metrics: Metrics) -> None:
        self.gateway = gateway
        self.prompts = prompts
        self.orders = orders
        self.metrics = metrics

    async def extract(self, ticket_text: str) -> ExtractionResult:
        template = self.prompts.get("extract_ticket")
        messages = template.render(
            ticket_text=ticket_text, schema=json.dumps(TICKET_SCHEMA, ensure_ascii=False)
        )
        resp = await self.gateway.generate(
            ModelRequest(
                messages,
                structured_output=StructuredOutputSpec("ticket_extraction", TICKET_SCHEMA),
                max_output_tokens=400,
                operation="extract",
            ),
            prompt_ref=template.ref,
        )
        base = {
            "finish_reason": resp.finish_reason,
            "prompt_ref": template.ref,
            "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        }

        # ---- 第 1 层：格式有效 ----
        if resp.refusal:
            return self._done(ExtractionResult("failed", None, "format", [f"refused: {resp.refusal}"], **base))
        if resp.finish_reason == "length":
            return self._done(ExtractionResult("failed", None, "format", ["output truncated"], **base))
        if resp.finish_reason == "content_filter":
            return self._done(ExtractionResult("failed", None, "format", ["blocked by content filter"], **base))
        try:
            data = json.loads(resp.text or "")
        except json.JSONDecodeError as exc:
            return self._done(ExtractionResult("failed", None, "format", [f"invalid JSON: {exc.msg}"], **base))

        # ---- 第 2 层：Schema 有效 ----
        schema_errors = [
            f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in _validator.iter_errors(data)
        ]
        if schema_errors:
            return self._done(ExtractionResult("failed", data, "schema", schema_errors, **base))

        # ---- 第 3 层：业务有效 ----
        issues = self._business_checks(data, ticket_text)
        if issues:
            # 应用强制要求人工复核，无论模型自己给出的 needs_human_review 是什么
            data["needs_human_review"] = True
        status: Status = "needs_review" if data["needs_human_review"] else "ok"
        return self._done(ExtractionResult(status, data, business_issues=issues, **base))

    def _business_checks(self, data: dict[str, Any], ticket_text: str) -> list[str]:
        issues: list[str] = []
        order_id = data["order_id"]
        if order_id is not None:
            if not ORDER_ID_FORMAT.match(order_id):
                issues.append("order_id_bad_format")
            elif order_id not in ticket_text:
                # 原文里根本没有这个订单号：典型的“编造”，格式再正确也不能用
                issues.append("order_id_not_in_source")
            elif not self.orders.exists(order_id):
                issues.append("order_not_found")
        elif data["category"] in ("退款", "物流"):
            issues.append("order_id_required_for_category")
        if not data["summary"].strip() or len(data["summary"]) > 120:
            issues.append("summary_length_invalid")
        return issues

    def _done(self, result: ExtractionResult) -> ExtractionResult:
        self.metrics.record_extraction(result.status)
        return result
