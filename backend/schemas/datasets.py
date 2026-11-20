"""Dataset upload payloads. API-only shapes: nothing here is written to storage."""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import StrictModel


class DatasetColumnPreview(StrictModel):
    """One column as it appears in the uploaded file, before any profiling has happened.

    This is a spelling and dtype aid for choosing the target column — not an analysis. The
    real profile comes from the EDA stage, which runs in the worker.
    """

    name: str
    dtype: str
    non_null_in_sample: int
    sample_values: list[str] = Field(default_factory=list, max_length=3)


class DatasetUploadResponse(StrictModel):
    """Where the uploaded file landed, and what is in it."""

    dataset_uri: str = Field(description="Pass this to POST /experiments as dataset_uri.")
    filename: str = Field(description="The file name as stored, sanitized if it had to be.")
    original_filename: str
    file_format: str
    size_bytes: int
    uploaded_at: datetime
    uploaded_by: str | None = None
    columns: list[DatasetColumnPreview] = Field(default_factory=list)
    preview_rows: int = Field(
        default=0, description="Rows read to build the column list, not the file's row count."
    )
    truncated_preview: bool = Field(
        default=False, description="True when the file has more rows than were previewed."
    )
