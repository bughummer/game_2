"""The Model_Version contract (control-ui-api design C4).

A Model_Version is a stable, listable handle to one trained model within a run — the current run or
an archived one — resolvable to its artifact, its metadata and the preprocessing it needs, so any
version can be loaded and validated later. It is a read model over storage that already exists
(``models/<name>/metadata.json`` for the current run, ``runs/<run_id>/models/<name>/metadata.json``
for archived runs); nothing is copied or overwritten to produce it.
"""

from datetime import datetime
from typing import Literal

from pydantic import Field

from ml_engine.contracts.common import ProblemType, StrictModel
from ml_engine.contracts.metrics import MetricSet

MODEL_VERSION_SCHEMA_VERSION = "1.0"

#: The run a version belongs to when it is the live (not yet archived) run.
CURRENT_RUN = "current"


class ArtifactRef(StrictModel):
    """Everything needed to load a Model_Version independently (design C4, Req 7.5)."""

    model_uri: str
    metadata_uri: str
    preprocessor_uri: str | None = Field(
        default=None, description="The fitted preprocessing the model consumes; None if unknown."
    )
    preprocessing_strategy: str | None = None


class ModelVersion(StrictModel):
    """A listable, independently-loadable handle to one trained model (design C4)."""

    schema_version: str = MODEL_VERSION_SCHEMA_VERSION
    version_id: str = Field(
        description="Stable: '<run_id>:<model>' for an archived run, 'current:<model>' for the "
        "live run."
    )
    experiment_id: str
    model_name: str
    display_name: str
    library: str
    problem_type: ProblemType
    hyperparameters: dict[str, float | int | str | bool | list[float] | None] = Field(
        default_factory=dict
    )
    primary_metric: str | None = None
    primary_score: float | None = None
    metrics: MetricSet = Field(default_factory=MetricSet)
    created_at: datetime | None = None
    source: Literal["training", "retrain", "grid_point", "free_form"] = "training"
    run_id: str | None = Field(
        default=None, description="The archived run id; None for the current run."
    )
    artifact_ref: ArtifactRef

    @property
    def is_current(self) -> bool:
        return self.run_id is None
