"""Grid-search contracts (grid-search design C2, C3).

A grid point is an ordinary ``ModelSpec`` with a distinct name and params, so most of the pipeline
is reused. These contracts add only what is new: a per-parameter *space* a plugin exposes so grid
values can be validated before submission, and the grid / plan shapes the service expands and shows
for approval.
"""

from typing import Literal

from pydantic import Field, model_validator

from ml_engine.contracts.common import StrictModel

GRID_SCHEMA_VERSION = "1.0"

#: The scalar types a searchable parameter (and a grid value) may take.
Scalar = float | int | str | bool

ParameterType = Literal["int", "float", "bool", "str"]


class ParameterSpec(StrictModel):
    """What values a searchable model parameter may take (grid-search Req 1.4).

    A plugin declares one of these per parameter it is sensible to grid-search. Numeric parameters
    carry optional bounds; categorical ones carry ``choices``. A value outside the space is
    rejected before submission, naming the parameter.
    """

    type: ParameterType
    minimum: float | int | None = Field(
        default=None, description="Inclusive lower bound (numeric)."
    )
    maximum: float | int | None = Field(
        default=None, description="Inclusive upper bound (numeric)."
    )
    choices: list[Scalar] | None = Field(default=None, description="Allowed values (categorical).")
    description: str | None = None

    def validate_value(self, value: Scalar) -> list[str]:
        """Return reasons ``value`` is invalid for this parameter (empty = valid)."""
        reasons: list[str] = []
        if self.type == "bool":
            if not isinstance(value, bool):
                reasons.append("must be a boolean")
            return reasons
        if self.type in ("int", "float"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                reasons.append(f"must be {self.type}")
                return reasons
            if self.type == "int" and not float(value).is_integer():
                reasons.append("must be an integer")
            if self.minimum is not None and value < self.minimum:
                reasons.append(f"must be >= {self.minimum}")
            if self.maximum is not None and value > self.maximum:
                reasons.append(f"must be <= {self.maximum}")
        if self.choices is not None and value not in self.choices:
            reasons.append(f"must be one of {self.choices}")
        return reasons


class ParameterGrid(StrictModel):
    """The user's grid: model name -> parameter name -> candidate values (grid-search Req 1.1)."""

    schema_version: str = GRID_SCHEMA_VERSION
    grid: dict[str, dict[str, list[Scalar]]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _non_empty_lists(self) -> "ParameterGrid":
        for model, params in self.grid.items():
            for name, values in params.items():
                if not values:
                    raise ValueError(f"{model}.{name} has no candidate values")
        return self

    @property
    def is_empty(self) -> bool:
        return not self.grid


class GridPoint(StrictModel):
    """One fully-specified parameter combination for one model."""

    model: str = Field(description="The base model name (a registered plugin).")
    point_id: str = Field(description="Stable path/job id: '<base>__<hash8>'.")
    display_name: str = Field(description="Readable '<base>#<hash8>'.")
    params: dict[str, Scalar] = Field(default_factory=dict)


class GridPlan(StrictModel):
    """The expanded grid shown for approval and recorded in the audit trail (grid-search Req 3)."""

    schema_version: str = GRID_SCHEMA_VERSION
    points: list[GridPoint]
    fold_count: int = Field(description="K in CV_Mode, else 1.")
    fits_per_point: int = Field(description="K + 1 in CV_Mode, else 1.")
    total_fits: int

    @property
    def point_count(self) -> int:
        return len(self.points)
