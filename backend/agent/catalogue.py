"""Tool catalogue for the ML Assistant (design C5).

Every ``backend/agent/tools_*.py`` module exposes ``register(registry, services)``. This module
finds those modules, imports them in name order and lets each register its tools into one
``ToolRegistry``. Splitting tools by module lets catalogue work proceed in parallel without two
changes touching the same file.

Tools reach data only through the service objects; this module imports no store implementation
and no boto client.
"""

import importlib
import pkgutil
from typing import Any

import backend.agent as _agent_package
from backend.agent.tools import ToolRegistry

TOOL_MODULE_PREFIX = "tools_"


class CatalogueError(RuntimeError):
    """A ``tools_*`` module cannot contribute to the registry (a programming error)."""


def tool_module_names() -> tuple[str, ...]:
    """Fully qualified names of the ``backend.agent.tools_*`` modules, sorted."""
    names = [
        f"{_agent_package.__name__}.{info.name}"
        for info in pkgutil.iter_modules(_agent_package.__path__)
        if info.name.startswith(TOOL_MODULE_PREFIX) and not info.ispkg
    ]
    return tuple(sorted(names))


def build_registry(services: Any) -> ToolRegistry:
    """Build the registry from every ``tools_*`` module.

    Raises ``CatalogueError`` when a module has no callable ``register``, and lets
    ``ToolRegistrationError`` from a module propagate (duplicate, reserved or unclassified tool).
    """
    registry = ToolRegistry()
    for name in tool_module_names():
        module = importlib.import_module(name)
        register = getattr(module, "register", None)
        if not callable(register):
            raise CatalogueError(f"tool module {name} has no register(registry, services)")
        register(registry, services)
    return registry
