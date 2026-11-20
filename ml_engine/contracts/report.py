"""Report document contracts (requirement 14).

A report is a *view* assembled from existing artifacts — the EDA report, the comparison, the
summary, the leakage findings. It computes no statistic of its own: every number is copied from
a deterministic artifact, so a report can never disagree with the artifacts it is built from.
The JSON report is this document; the PDF is a rendering of it.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from ml_engine.contracts.common import StrictModel

REPORT_SCHEMA_VERSION = "1.0"


class ReportKind(StrEnum):
    EDA = "eda"
    MODEL_COMPARISON = "model_comparison"


class ReportSection(StrictModel):
    """One titled section. ``rows`` is a small table; ``notes`` are plain-language lines."""

    title: str
    summary: str | None = None
    rows: list[dict[str, str]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Report(StrictModel):
    schema_version: str = REPORT_SCHEMA_VERSION
    experiment_id: str
    kind: ReportKind
    title: str
    generated_at: datetime
    sections: list[ReportSection] = Field(default_factory=list)
    #: Keys of the artifacts this report was assembled from — provenance, not new data.
    source_artifacts: list[str] = Field(default_factory=list)
    #: Where an LLM narrative, if any, came from. The narrative itself is grounded in the
    #: sections above and adds no numbers (requirement 14.4). Empty when no LLM was used.
    narrative: str | None = None
    narrative_source: str | None = None
