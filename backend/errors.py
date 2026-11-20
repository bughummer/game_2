"""One error model for the whole control plane.

Framework-free (no FastAPI): the MCP server and any future HTTP layer both map these to their
own transport. Every expected failure carries a stable ``code`` and a user-safe ``message``;
stack traces go to logs, never to a caller.

Ported from old_repo/backend/errors.py, trimmed to the exception hierarchy (the FastAPI handlers
are not ported — the MCP layer does its own mapping).
"""

from typing import Any


class AppError(Exception):
    """Base class for every expected failure. ``code`` is stable and safe to show a user."""

    code = "INTERNAL_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ValidationError(AppError):
    code = "VALIDATION_ERROR"


class NotFoundError(AppError):
    code = "NOT_FOUND"


class ConflictError(AppError):
    """The request is valid but the experiment is not in a state that allows it."""

    code = "INVALID_STATE"


class ArtifactNotReadyError(AppError):
    """The artifact does not exist yet because its stage has not completed."""

    code = "ARTIFACT_NOT_READY"


class AuthorizationError(AppError):
    """The caller is not allowed to perform this operation on this experiment."""

    code = "NOT_AUTHORIZED"


class UpstreamError(AppError):
    """An AWS dependency failed."""

    code = "UPSTREAM_ERROR"


class FeatureDisabledError(AppError):
    """A capability was requested that is switched off in this deployment (e.g. Bedrock)."""

    code = "FEATURE_DISABLED"
