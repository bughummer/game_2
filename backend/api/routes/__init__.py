"""REST resource modules (control-ui-api design C3).

Each module exposes a ``routes()`` collector returning ``(method, path_suffix, handler_factory)``
tuples that ``backend.api.app._collect_routes`` aggregates and mounts under ``/api``. A module holds
only thin wrappers: resolve the caller, enforce ownership, call exactly the ``backend.services``
method the matching MCP tool calls, audit the mutation. No ML or business logic lives here.
"""
