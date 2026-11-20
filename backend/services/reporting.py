"""Build and store downloadable reports (requirement 14): JSON always, PDF when available.

The service assembles a ``Report`` from the stored artifacts, writes the machine-readable JSON
and a self-contained HTML rendering to S3, and — if WeasyPrint is installed — a PDF. WeasyPrint
is an optional dependency evaluated per the design (mature, HTML→PDF, no browser); when it is
absent the service still produces JSON + HTML and says so, rather than failing.

Reports are a view over artifacts and add no numbers. An optional LLM narrative may be supplied
by the caller; it is stored as prose grounded in the sections, never as new statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.config import Settings
from backend.errors import ArtifactNotReadyError
from backend.repositories import ExperimentRepository
from ml_engine.contracts.comparison import ExperimentSummary
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.report import Report, ReportKind
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    read_model_if_exists,
    write_model,
)
from ml_engine.reporting.documents import build_comparison_report, build_eda_report
from ml_engine.reporting.render import render_html

LOGGER = logging.getLogger("ml_factory.services.reporting")


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    kind: ReportKind
    json_uri: str
    html_uri: str
    pdf_uri: str | None  # None when WeasyPrint is not installed
    pdf_available: bool


class ReportingService:
    def __init__(
        self, *, settings: Settings, repository: ExperimentRepository, store: ObjectStore
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._store = store

    def build_eda_report(
        self,
        experiment_id: str,
        *,
        narrative: str | None = None,
        narrative_source: str | None = None,
    ) -> ReportArtifacts:
        layout = self._layout(experiment_id)
        eda = read_model_if_exists(self._store, layout.eda, EdaReport)
        if eda is None:
            raise ArtifactNotReadyError("EDA has not completed, so no EDA report can be built.")
        leakage = read_model_if_exists(self._store, layout.leakage, LeakageReport)
        report = build_eda_report(
            eda=eda, leakage=leakage, narrative=narrative, narrative_source=narrative_source
        )
        return self._store_report(layout, report)

    def build_comparison_report(
        self,
        experiment_id: str,
        *,
        narrative: str | None = None,
        narrative_source: str | None = None,
    ) -> ReportArtifacts:
        layout = self._layout(experiment_id)
        summary = read_model_if_exists(self._store, layout.summary, ExperimentSummary)
        if summary is None:
            raise ArtifactNotReadyError(
                "The experiment summary is not available yet; training must complete first."
            )
        report = build_comparison_report(
            summary=summary, narrative=narrative, narrative_source=narrative_source
        )
        return self._store_report(layout, report)

    def get_report(self, experiment_id: str, kind: ReportKind) -> Report | None:
        layout = self._layout(experiment_id)
        return read_model_if_exists(self._store, layout.report_document(kind.value, "json"), Report)

    # --- internals ----------------------------------------------------------------------------
    def _store_report(self, layout: ExperimentLayout, report: Report) -> ReportArtifacts:
        kind = report.kind
        json_uri = layout.report_document(kind.value, "json")
        html_uri = layout.report_document(kind.value, "html")
        write_model(self._store, json_uri, report)
        html = render_html(report)
        self._store.write_bytes(html_uri, html.encode("utf-8"))

        pdf_uri: str | None = None
        pdf = _render_pdf(html)
        if pdf is not None:
            pdf_uri = layout.report_document(kind.value, "pdf")
            self._store.write_bytes(pdf_uri, pdf)
        else:
            LOGGER.info("WeasyPrint not installed; wrote JSON + HTML only for %s", kind.value)
        return ReportArtifacts(
            kind=kind,
            json_uri=json_uri,
            html_uri=html_uri,
            pdf_uri=pdf_uri,
            pdf_available=pdf_uri is not None,
        )

    def _layout(self, experiment_id: str) -> ExperimentLayout:
        return ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)


def _render_pdf(html: str) -> bytes | None:
    """Render HTML to PDF with WeasyPrint if it is installed; otherwise return None.

    WeasyPrint is optional so the default test suite and the minimal control-plane image need no
    native rendering stack. The import is lazy for the same reason.
    """
    try:
        from weasyprint import HTML  # type: ignore
    except Exception:
        return None
    try:
        return HTML(string=html).write_pdf()
    except Exception:  # a rendering failure must not take down report generation
        LOGGER.exception("WeasyPrint failed to render a report PDF")
        return None
