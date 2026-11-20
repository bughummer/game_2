"""Composition root: build the control-plane services from settings.

One place wires the S3 store, the AWS clients, the orchestrator, the status monitor and the use
cases, so a deployment is configured from ``Settings`` alone and tests can substitute any seam.
"""

from dataclasses import dataclass
from typing import Any

from backend.agent.converse import ConverseClient
from backend.audit import AuditLog
from backend.auth import Authorizer
from backend.config import Settings
from backend.governance import GovernancePolicy
from backend.orchestration import SageMakerOrchestrator
from backend.pricing import PriceTable
from backend.repositories import ExperimentRepository
from backend.services import (
    AnalysisService,
    DatasetService,
    ExperimentService,
    FeatureProposalService,
    FreeFormService,
    LeakageService,
    ModelVersionService,
    ReasoningService,
    ReportingService,
    ValidationService,
)
from backend.tracking import ExecutionPoller, StatusMonitor
from ml_engine.io import ObjectStore, S3ObjectStore
from ml_engine.reasoning import BedrockReasoningClient, ReasoningClient


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    store: ObjectStore
    experiments: ExperimentService
    datasets: DatasetService
    leakage: LeakageService
    reporting: ReportingService
    proposals: FeatureProposalService
    reasoning: ReasoningService
    analysis: AnalysisService
    model_versions: ModelVersionService
    validation: ValidationService
    free_form: FreeFormService
    orchestrator: SageMakerOrchestrator
    monitor: StatusMonitor
    poller: ExecutionPoller
    authorizer: Authorizer
    audit: AuditLog
    governance: GovernancePolicy
    # The ML Assistant (backend.agent.service.AgentService); typed loosely so this module stays
    # importable without the agent package's types at annotation time.
    agent: Any = None


def build_container(
    settings: Settings,
    *,
    store: ObjectStore | None = None,
    orchestrator: Any = None,
    sagemaker_client: Any = None,
    logs_client: Any = None,
    s3_client: Any = None,
    reasoning_client: ReasoningClient | None = None,
    converse_client: ConverseClient | None = None,
) -> AppContainer:
    """Build the services. In production the clients come from ``backend.aws``; tests inject them.

    ``store``/``orchestrator`` may be supplied directly (tests use the in-memory store and the
    fake SageMaker); otherwise they are built from ``settings``.

    ``converse_client`` is the ML Assistant's Bedrock Converse client. When omitted, the agent
    wiring builds a lazily connecting Bedrock client only if ``agent_enabled`` is true, and a
    client that refuses every request otherwise. Building the agent opens no connection and
    starts no thread.
    """
    problems = settings.problems()
    if problems and store is None:
        raise ValueError("Configuration is incomplete: " + "; ".join(problems))

    if store is None or s3_client is None or sagemaker_client is None:
        built = _build_aws(settings)
        s3_client = s3_client or built["s3"]
        sagemaker_client = sagemaker_client or built["sagemaker"]
        logs_client = logs_client or built["logs"]
        store = store or S3ObjectStore(
            s3_client,
            server_side_encryption=settings.s3_sse,
            kms_key_id=settings.s3_kms_key_id,
        )

    prices = PriceTable.load(settings.price_table_path) if settings.enforce_budget else None
    orchestrator = orchestrator or SageMakerOrchestrator(settings, store, sagemaker_client, prices)
    monitor = StatusMonitor(orchestrator, store, sagemaker_client, logs_client=logs_client)
    # Built always; only the server process starts its thread (mcp_server/app.py), so tests and
    # scripts that build a container never spawn a background poller.
    poller = ExecutionPoller(
        monitor,
        store,
        settings.artifact_root,
        interval_seconds=settings.status_poll_interval_seconds,
        rediscover_seconds=settings.status_rediscover_interval_seconds,
    )
    register = getattr(orchestrator, "on_started", None)
    if callable(register):
        register(poller.register)
    repository = ExperimentRepository(settings=settings, store=store)
    authorizer = Authorizer(repository, dev_identity=settings.dev_identity)
    audit = AuditLog(store, enabled=settings.audit_log_enabled)
    governance = GovernancePolicy.from_settings(settings)
    experiments = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    datasets = DatasetService(settings=settings, store=store, s3_client=s3_client)
    reporting = ReportingService(settings=settings, repository=repository, store=store)
    if reasoning_client is None and settings.bedrock_enabled and settings.bedrock_model_id:
        reasoning_client = BedrockReasoningClient(
            _build_bedrock(settings),
            model_id=settings.bedrock_model_id,
            max_tokens=settings.bedrock_max_tokens,
        )
    reasoning = ReasoningService(
        settings=settings,
        repository=repository,
        store=store,
        client=reasoning_client,
        governance=governance,
    )
    leakage = LeakageService(
        settings=settings, repository=repository, store=store, reasoning=reasoning
    )
    analysis = AnalysisService(
        settings=settings,
        repository=repository,
        store=store,
        orchestrator=orchestrator,
        experiments=experiments,
    )
    proposals = FeatureProposalService(
        settings=settings,
        repository=repository,
        store=store,
        client=reasoning_client,
        orchestrator=orchestrator,
    )
    model_versions = ModelVersionService(settings=settings, repository=repository, store=store)
    validation = ValidationService(
        settings=settings,
        repository=repository,
        store=store,
        orchestrator=orchestrator,
        experiments=experiments,
    )
    free_form = FreeFormService(
        settings=settings,
        repository=repository,
        store=store,
        orchestrator=orchestrator,
        experiments=experiments,
        client=reasoning_client,
    )
    container = AppContainer(
        settings=settings,
        store=store,
        experiments=experiments,
        datasets=datasets,
        leakage=leakage,
        reporting=reporting,
        proposals=proposals,
        reasoning=reasoning,
        analysis=analysis,
        model_versions=model_versions,
        validation=validation,
        free_form=free_form,
        orchestrator=orchestrator,
        monitor=monitor,
        poller=poller,
        authorizer=authorizer,
        audit=audit,
        governance=governance,
    )
    # Imported here so the agent package loads only when a container is built. Tool handlers
    # reach the platform services through the container itself (``ctx.services``).
    from backend.agent.wiring import build_agent_service

    container.agent = build_agent_service(
        settings,
        store=store,
        services=container,
        converse_client=converse_client,
        audit_log=audit,
        authorizer=authorizer,
        governance=governance,
    )
    return container


def _build_aws(settings: Settings) -> dict[str, Any]:
    from backend.aws import build_client

    return {
        "s3": build_client("s3", settings),
        "sagemaker": build_client("sagemaker", settings),
        "logs": build_client("logs", settings),
    }


def _build_bedrock(settings: Settings) -> Any:
    from backend.aws import build_client

    return build_client("bedrock-runtime", settings)
