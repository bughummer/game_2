"""Grid expansion (grid-search design C4), pure and offline.

Turns a ``ParameterGrid`` into one ``GridPoint`` per parameter combination. A grid point is an
ordinary ``ModelSpec`` downstream; this module only computes the points and their stable ids. No
IO, no plugin construction beyond the parameter-space validation the caller supplies.
"""

from itertools import product

from ml_engine.contracts.grid import GridPoint, ParameterGrid
from ml_engine.models.registry import GRID_POINT_SEPARATOR, grid_point_id


class GridExpansionError(ValueError):
    """A grid value or key is invalid. ``reason`` is stable; ``details`` name the offender."""

    def __init__(self, message: str, reason: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details or {}


def expand_model_grid(base: str, params_grid: dict[str, list]) -> list[GridPoint]:
    """The cross-product of one model's per-parameter value lists, as ``GridPoint``s.

    An empty grid for the model yields no points here (the caller adds a single default point for a
    model with no grid entry). Each point's id is stable for its exact combination.
    """
    if not params_grid:
        return []
    names = sorted(params_grid)
    combos = product(*(params_grid[name] for name in names))
    points: list[GridPoint] = []
    for combo in combos:
        values = dict(zip(names, combo, strict=True))
        point_id = grid_point_id(base, values)
        display = point_id.replace(GRID_POINT_SEPARATOR, "#", 1)
        points.append(GridPoint(model=base, point_id=point_id, display_name=display, params=values))
    return points


def expand_grid(parameter_grid: ParameterGrid, base_names: list[str]) -> dict[str, list[GridPoint]]:
    """Grid points per base model name, for the models named in ``base_names``.

    A model in ``base_names`` with no grid entry maps to an empty list (the caller trains it once
    with its defaults, Req 1.5). A grid entry for a model not in ``base_names`` is the caller's to
    reject; this function only expands.
    """
    result: dict[str, list[GridPoint]] = {}
    for base in dict.fromkeys(base_names):
        result[base] = expand_model_grid(base, parameter_grid.grid.get(base, {}))
    return result
