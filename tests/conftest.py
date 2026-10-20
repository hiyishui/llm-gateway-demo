"""测试公共夹具：全部使用离线 Mock 模型服务，不读取 .env，不访问网络。"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from llm_gateway.adapters.mock_server import MockLLMServer
from llm_gateway.api import create_app
from llm_gateway.config import Secret, Settings
from llm_gateway.service import GatewayService

ALICE = {"Authorization": "Bearer t-alice"}
BOB = {"Authorization": "Bearer t-bob"}
FAKE_KEY = "sk-test-SHOULD-NEVER-BE-LOGGED-123"


def make_settings(**overrides) -> Settings:
    base = Settings(
        llm_provider="mock",
        llm_api_key=Secret(FAKE_KEY),
        llm_model="mock-model",
        llm_retry_base_delay=0.001,
        llm_retry_max_delay=0.005,
        gateway_tokens={"t-alice": "alice", "t-bob": "bob"},
        tool_timeout_seconds=0.2,
        log_level="INFO",
    )
    return replace(base, **overrides)


def fast_mock() -> MockLLMServer:
    return MockLLMServer(latency_ms=(0, 0), chunk_delay=0)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings, fast_mock().transport())) as c:
        yield c


@pytest.fixture
async def service(settings: Settings):
    svc = GatewayService(settings, fast_mock().transport())
    svc.orders.slow_seconds = 1.0
    yield svc
    await svc.aclose()
