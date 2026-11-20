"""Batch scoring: apply a trained model to a new file and write predictions to S3.

No endpoint is created (requirement 4.6). A SageMaker Processing job loads the experiment's
fitted preprocessing pipeline and model, computes the approved derived features the same way
preparation did, and writes ``scoring/<id>/predictions.csv`` plus this summary. Nothing is
re-fitted, so the scored file cannot influence the model.
"""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import StrictModel
from ml_engine.contracts.warnings import AnalysisWarning

SCORING_SCHEMA_VERSION = "1.0"


class ScoringRequest(StrictModel):
    schema_version: str = SCORING_SCHEMA_VERSION
    scoring_id: str
    experiment_id: str
    model_name: str
    dataset_uri: str
    source_uri: str | None = None
    file_format: str
    #: Columns copied verbatim from the input to the output, so rows can be joined back.
    passthrough_columns: list[str] = Field(default_factory=list)
    requested_at: datetime
    requested_by: str | None = None
    execution_id: str | None = None


class ScoringReport(StrictModel):
    schema_version: str = SCORING_SCHEMA_VERSION
    scoring_id: str
    experiment_id: str
    model_name: str
    generated_at: datetime
    dataset_uri: str
    predictions_uri: str
    row_count: int
    prediction_columns: list[str] = Field(default_factory=list)
    passthrough_columns: list[str] = Field(default_factory=list)
    #: Categorical distribution (classification) or summary stats (regression) of predictions.
    prediction_summary: dict[str, float] = Field(default_factory=dict)
    image_uri: str | None = None
    training_image_uri: str | None = None
    warnings: list[AnalysisWarning] = Field(default_factory=list)
