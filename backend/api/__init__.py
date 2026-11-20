"""The HTTP REST API — a peer of the MCP server over the one service layer.

Thin Starlette routes that resolve the caller, enforce ownership, call a ``backend.services``
function, map ``AppError`` to an HTTP status, and audit mutations. No ML or business logic lives
here (the same rule the MCP follows).
"""

from backend.api.app import API_PREFIX, register_api_routes

__all__ = ["API_PREFIX", "register_api_routes"]
