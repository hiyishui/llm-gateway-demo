"""LLM Gateway Demo：配合《大模型应用基础》的综合练习——一个小型模型网关。"""


def main() -> None:
    """命令行入口：`uv run llm-gateway`，监听地址由 HOST / PORT 环境变量决定。"""
    import os

    import uvicorn
    from dotenv import load_dotenv

    load_dotenv(".env")
    uvicorn.run(
        "llm_gateway.api:create_app",
        factory=True,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        # 访问日志可能包含 URL 参数，这里关闭，统一使用网关自己的脱敏 JSON 日志
        access_log=False,
    )
