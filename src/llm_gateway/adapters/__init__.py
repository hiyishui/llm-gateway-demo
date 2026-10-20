"""模型适配器：应用内部统一类型 ⇄ 供应商协议。"""

from .base import ModelAdapter
from .openai_compat import OpenAICompatAdapter

__all__ = ["ModelAdapter", "OpenAICompatAdapter"]
