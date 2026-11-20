"""Assemble the ``AgentService`` from settings, an object store and the platform services.

``build_agent_service`` opens no connection and writes nothing: the session store, audit
writer, tool registry, executor and loop are plain objects, the default runner's thread pool
starts no thread until the first kick, and the Bedrock client (when built here) creates its
boto3 client on first use only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from backend.agent.audit import AgentAudit
from backend.agent.catalogue import build_registry
from backend.agent.converse import (
    BedrockConverseClient,
    ConverseClient,
    ConverseMessage,
    ConverseRequestError,
    ConverseTurn,
    ToolDefinition,
)
from backend.agent.estimate import CostEstimator
from backend.agent.execute import ToolExecutor
from backend.agent.inputs import ToolInputs
from backend.agent.loop import AgentLoop, LoopSettings
from backend.agent.runner import ThreadAgentRunner
from backend.agent.service import AgentService
from backend.agent.store import SessionStore
from backend.agent.watcher import AgentJobWatcher
from backend.audit import AuditLog
from backend.auth import Authorizer
from backend.governance import GovernancePolicy
from backend.repositories import ExperimentRepository
from ml_engine.io.object_store import ObjectStore

LOGGER = logging.getLogger("ml_factory.agent.wiring")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class UnavailableConverseClient:
    """Stands in when the agent is disabled and no client was injected. Never calls out."""

    model_id = ""

    def converse(
        self,
        *,
        system: str,
        messages: Sequence[ConverseMessage],
        tools: Sequence[ToolDefinition],
        max_tokens: int,
    ) -> ConverseTurn:
        del system, messages, tools, max_tokens
        raise ConverseRequestError("the ML Assistant is disabled (ML_FACTORY_AGENT_ENABLED)")


def _estimator(settings: Any) -> CostEstimator | None:
    """The Guards' cost estimator. Without one every estimate is "no estimate available"."""
    try:
        return CostEstimator(settings, None)
    except Exception:  # an unusable estimator fails safe: approval, never unattended spend
        LOGGER.exception("Cost estimator could not be built; estimates are unavailable")
        return None


def default_converse_client(settings: Any) -> ConverseClient:
    """A lazily connecting Bedrock client when enabled, otherwise a client that refuses."""
    if not settings.agent_enabled:
        return UnavailableConverseClient()
    return BedrockConverseClient.from_settings(settings)


def build_agent_service(
    settings: Any,
    *,
    store: ObjectStore,
    services: Any = None,
    converse_client: ConverseClient | None = None,
    audit_log: AuditLog | None = None,
    authorizer: Authorizer | None = None,
    governance: GovernancePolicy | None = None,
    runner: Any = None,
    clock: Callable[[], datetime] | None = None,
) -> AgentService:
    """Build the service and its loop. Every collaborator may be injected (tests, container).

    ``services`` is what tool handlers reach through ``ctx.services`` (at least ``.datasets``
    and ``.experiments``); the container passes itself or a namespace of its services.
    ``runner`` defaults to a ``ThreadAgentRunner``; a runner with ``bind`` is bound to the loop.
    """
    now = clock or _utcnow
    root = settings.artifact_root
    services = services if services is not None else SimpleNamespace()
    audit_log = audit_log or AuditLog(store, enabled=settings.audit_log_enabled)
    if authorizer is None:
        repository = ExperimentRepository(settings=settings, store=store)
        authorizer = Authorizer(repository, dev_identity=settings.dev_identity)
    governance = governance or GovernancePolicy.from_settings(settings)
    converse = converse_client if converse_client is not None else default_converse_client(settings)

    sessions = SessionStore(store, root, lease_seconds=settings.agent_lease_seconds, clock=now)
    audit = AgentAudit(audit_log, root)
    allowed = [*settings.allowed_instance_types, *settings.allowed_free_form_instance_types]
    inputs = ToolInputs(audit=audit, authorizer=authorizer, allowed_instance_types=allowed)
    registry = build_registry(services)
    executor = ToolExecutor(inputs=inputs, services=services, clock=now)
    loop = AgentLoop(
        store=sessions,
        converse=converse,
        registry=registry,
        executor=executor,
        inputs=inputs,
        audit=audit,
        governance=governance,
        settings=LoopSettings.from_settings(settings),
        services=services,
        clock=now,
        estimator=_estimator(settings),
    )
    if runner is None:
        runner = ThreadAgentRunner(max_workers=settings.agent_max_concurrent_sessions)
    bind = getattr(runner, "bind", None)
    if callable(bind):
        bind(loop)
    # Built always, started only by the server (mcp_server/app.py) when the agent is enabled.
    watcher = AgentJobWatcher(
        sessions,
        getattr(services, "analysis", None),
        interval_seconds=getattr(settings, "agent_watch_interval_seconds", 15),
        clock=now,
    )
    loop.on_waiting_for_job = watcher.register
    service = AgentService(
        settings=settings,
        store=sessions,
        runner=runner,
        audit=audit,
        loop=loop,
        clock=now,
        registry=registry,
        inputs=inputs,
        analysis=getattr(services, "analysis", None),
        orchestrator=getattr(services, "orchestrator", None),
        services=services,
    )
    service.watcher = watcher  # type: ignore[attr-defined]
    return service
