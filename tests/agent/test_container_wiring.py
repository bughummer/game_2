"""The container exposes the ML Assistant offline (ml-assistant task 4.16, design C20).

Requirements 13.5, 15.8: ``build_container(..., converse_client=fake)`` builds
``container.agent`` with no network, no thread, and the platform services as ``ctx.services``.
"""

from __future__ import annotations

from backend.agent.service import AgentService
from backend.agent.wiring import UnavailableConverseClient
from backend.config import Settings
from backend.container import build_container
from tests.backend.conftest import IMAGE, ROLE
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.fake_sagemaker import FakeLogs, FakeSageMaker
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.scripted_converse import ScriptedFakeConverseClient


def _settings(price_table_path, **overrides) -> Settings:
    return Settings(
        artifact_root=ARTIFACT_ROOT,
        sagemaker_execution_role_arn=ROLE,
        stage_image_uri=IMAGE,
        price_table_path=price_table_path,
        **overrides,
    )


def _build(settings: Settings, store: InMemoryObjectStore, **kwargs):
    return build_container(
        settings,
        store=store,
        sagemaker_client=FakeSageMaker(),
        s3_client=object(),  # never called: no dataset operation runs here
        logs_client=FakeLogs(),
        **kwargs,
    )


def test_container_exposes_agent_with_the_injected_client_offline(price_table_path):
    store = InMemoryObjectStore()
    fake = ScriptedFakeConverseClient(steps=())
    container = _build(_settings(price_table_path, agent_enabled=True), store, converse_client=fake)
    try:
        assert isinstance(container.agent, AgentService)
        loop = container.agent._loop
        assert loop._converse is fake
        assert loop._services is container
        # Building opens no connection, makes no request and writes nothing.
        assert fake.calls == 0
        assert list(store._objects) == []
    finally:
        container.agent._runner.shutdown()


def test_disabled_agent_without_a_client_refuses_instead_of_calling_bedrock(price_table_path):
    container = _build(_settings(price_table_path), InMemoryObjectStore())
    try:
        assert isinstance(container.agent._loop._converse, UnavailableConverseClient)
    finally:
        container.agent._runner.shutdown()
