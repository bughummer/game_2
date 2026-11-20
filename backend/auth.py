"""Authorization boundary (requirement 15.1, 15.2).

Experiment ownership and user-scoped access are enforced here. The *authentication* mechanism —
how a trustworthy identity is established — is deferred until the LibreChat integration is real
(handover decisions 10-12). Until then a single configured development identity is used and the
server logs, loudly, that it is not a production auth boundary.

The rule the rest of the system relies on: a mutation always carries a ``Principal``, and
``require_owner`` is called before any state change on an experiment.
"""

import logging
from dataclasses import dataclass

from backend.errors import AuthorizationError
from backend.repositories import ExperimentRepository

LOGGER = logging.getLogger("ml_factory.auth")


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is making a request. ``trusted`` is False for the dev stand-in."""

    user: str
    trusted: bool = False


class Authorizer:
    def __init__(
        self, repository: ExperimentRepository, *, dev_identity: str | None = None
    ) -> None:
        self._repository = repository
        self._dev_identity = dev_identity
        if dev_identity:
            LOGGER.warning(
                "Using the development identity %r. This is NOT a production auth boundary; "
                "wire real authentication before exposing the MCP beyond local use.",
                dev_identity,
            )

    def principal(self, claimed_user: str | None) -> Principal:
        """Resolve the caller.

        A verified identity would arrive from the authenticated transport. Until that exists the
        configured dev identity is used; a header value is never trusted on its own.
        """
        if self._dev_identity:
            return Principal(user=self._dev_identity, trusted=False)
        if not claimed_user:
            raise AuthorizationError("No authenticated identity is available for this request.")
        # No verified auth boundary is configured and no dev identity is set: refuse rather than
        # trust a raw claim.
        raise AuthorizationError(
            "This deployment has no verified authentication configured, so mutations are refused."
        )

    def require_owner(self, principal: Principal, experiment_id: str) -> None:
        owner = self._repository.owner(experiment_id)
        if owner is not None and owner != principal.user:
            raise AuthorizationError(
                "This experiment belongs to another user.",
                {"experiment_id": experiment_id},
            )
