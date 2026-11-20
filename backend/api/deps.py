"""Request-scoped helpers for the REST API (control-ui-api design C1).

Resolving the caller and auditing a mutation go through the same ``Authorizer`` and ``AuditLog``
the MCP uses. A header is only a *claim*; the ``Authorizer`` decides (today: the configured dev
identity, flagged). Real authentication is wired at the same point the MCP's is.
"""

from starlette.requests import Request

from backend.auth import Principal
from backend.container import AppContainer

#: A claimed identity header. Never trusted on its own — passed to Authorizer.principal, which
#: ignores it unless a verified auth boundary is configured (matching the MCP's ``user`` claim).
USER_HEADER = "X-ML-Factory-User"


def claimed_user(request: Request) -> str | None:
    return request.headers.get(USER_HEADER)


def principal(container: AppContainer, request: Request) -> Principal:
    """Resolve who is calling, exactly as the MCP does."""
    return container.authorizer.principal(claimed_user(request))


def require_owner(container: AppContainer, who: Principal, experiment_id: str) -> None:
    container.authorizer.require_owner(who, experiment_id)


def audit_mutation(
    container: AppContainer,
    experiment_id: str,
    action: str,
    actor: str | None,
    **detail: str,
) -> None:
    """Audit one mutation on an existing experiment (reads are never audited)."""
    from backend.errors import AppError

    try:
        prefix = container.experiments.get(experiment_id).artifact_prefix
    except AppError:
        return
    container.audit.record(
        artifact_prefix=prefix,
        experiment_id=experiment_id,
        action=action,
        actor=actor,
        detail={k: str(v) for k, v in detail.items()},
    )
