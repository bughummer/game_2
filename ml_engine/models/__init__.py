"""Model plugins: one interface, many libraries, no model-specific orchestration code."""

from ml_engine.models.base import (
    ModelNotAvailableError,
    ModelPlugin,
    TrainingContext,
    normalize_importance,
)
from ml_engine.models.registry import (
    GRID_POINT_SEPARATOR,
    UnknownModelError,
    all_plugins,
    base_model_name,
    build_ensemble_plugin,
    catalogue,
    default_model_names,
    get_plugin,
    grid_point_id,
    is_ensemble_name,
    plugins_for,
    register,
    resolve_plugin,
)

__all__ = [
    "GRID_POINT_SEPARATOR",
    "ModelNotAvailableError",
    "ModelPlugin",
    "TrainingContext",
    "UnknownModelError",
    "all_plugins",
    "base_model_name",
    "build_ensemble_plugin",
    "catalogue",
    "default_model_names",
    "get_plugin",
    "grid_point_id",
    "is_ensemble_name",
    "normalize_importance",
    "plugins_for",
    "register",
    "resolve_plugin",
]
