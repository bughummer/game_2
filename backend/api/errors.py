"""Map the framework-free ``AppError`` hierarchy to HTTP (control-ui-api design C2).

The REST layer translates the stable ``AppError.code`` to a status, exactly as the MCP's ``_safe``
translates it to a ``ToolError``. Any non-``AppError`` becomes a generic 500 with no stack trace.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.errors import AppError

LOGGER = logging.getLogger("ml_factory.api")

#: Stable AppError.code -> HTTP status. Codes come from backend/errors.py.
STATUS_BY_CODE: dict[str, int] = {
    "VALIDATION_ERROR": 400,
    "NOT_AUTHORIZED": 403,
    "FEATURE_DISABLED": 403,
    "NOT_FOUND": 404,
    "INVALID_STATE": 409,
    "ARTIFACT_NOT_READY": 409,
    "UPSTREAM_ERROR": 502,
    "INTERNAL_ERROR": 500,
}
DEFAULT_STATUS = 500


def status_for(error: AppError) -> int:
    return STATUS_BY_CODE.get(error.code, DEFAULT_STATUS)


def error_response(error: AppError) -> JSONResponse:
    return JSONResponse(error.as_dict(), status_code=status_for(error))


def generic_error_response() -> JSONResponse:
    """A 500 that never leaks internal detail or a stack trace."""
    return JSONResponse(
        {"code": "INTERNAL_ERROR", "message": "An internal error occurred.", "details": {}},
        status_code=500,
    )


Handler = Callable[[Request], Awaitable[JSONResponse]]


def safe(handler: Handler) -> Handler:
    """Wrap a route so an ``AppError`` maps to its status and nothing else leaks.

    The MCP wraps every tool with an equivalent ``_safe``; this is its HTTP sibling.
    """

    async def wrapped(request: Request) -> JSONResponse:
        try:
            return await handler(request)
        except AppError as error:
            return error_response(error)
        except Exception:  # never leak a trace to the client
            LOGGER.exception("Unhandled error in %s", getattr(handler, "__name__", "route"))
            return generic_error_response()

    wrapped.__name__ = getattr(handler, "__name__", "route")
    return wrapped


def json_ok(payload: Any, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code)
