"""Derived-feature profiling job: profile and leakage-screen the approved derived features.

Runs as a SageMaker Processing job after a person approves derived features, so the control plane
never loads the dataset. It reads the approval (``config/feature_proposals.json``), computes the
approved columns over the full dataset, profiles them and screens them for leakage, and writes
``config/derived_profiles.json`` for the feature-review table.

Leak-safe for the same reason the preparation job is: every operation in the vocabulary is
row-wise and nothing is fitted here, so profiling before the split moves no information across
the fold boundary. The authoritative computation still happens in preparation at training time;
this job only gives the reviewer real statistics before then.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime

from jobs._common.runtime import JobError, base_parser, run_entrypoint
from ml_engine.contracts.common import ProblemType, RequestedProblemType
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.experiment import ExperimentDefinition
from ml_engine.contracts.proposals import (
    DerivedFeatureStatus,
    DerivedProfileReport,
    FeatureProposalReport,
)
from ml_engine.features import (
    build_derived_profiles,
    column_types_from_frame,
    derive_and_screen,
    validate_proposals,
)
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    load_dataset,
    read_model,
    read_model_if_exists,
    write_model,
)

LOGGER = logging.getLogger("ml_factory.jobs.derived_profiling")

FAILURE_DIR = "derived"


def run_derived_profiling(
    store: ObjectStore, layout: ExperimentLayout, *, experiment_id: str
) -> DerivedProfileReport:
    """Profile the currently approved derived features. Writes and returns the report."""
    definition = read_model(store, layout.definition, ExperimentDefinition)
    proposals = read_model_if_exists(store, layout.feature_proposals, FeatureProposalReport)
    approved = proposals.approved if proposals is not None else []
    if not approved:
        return _write(store, layout, _empty(definition))

    LOGGER.info("Loading dataset %s", definition.dataset.uri)
    dataset = load_dataset(store, definition.dataset.uri, definition.dataset.file_format)
    frame = dataset.frame
    if definition.target_column not in frame.columns:
        raise JobError(
            "TARGET_COLUMN_MISSING",
            f"Target column '{definition.target_column}' is not present in the dataset.",
        )
    frame = frame[frame[definition.target_column].notna()]

    # Re-validate against the data as it is now, exactly as preparation will.
    accepted, rejected = validate_proposals(
        approved, column_types_from_frame(frame), target_column=definition.target_column
    )
    for refusal in rejected:
        LOGGER.warning("Approved feature %s cannot be built: %s", refusal.name, refusal.message)

    _frame, derived, subset_eda, leakage = derive_and_screen(
        frame,
        accepted,
        experiment_id=experiment_id,
        target_column=definition.target_column,
        problem_type=_problem_type(store, layout, definition),
        source_uri=definition.dataset.uri,
        file_format=dataset.file_format,
    )
    report = build_derived_profiles(
        derived,
        subset_eda,
        leakage,
        experiment_id=experiment_id,
        target_column=definition.target_column,
        status=DerivedFeatureStatus.APPROVED,
    )
    LOGGER.info("Profiled %d approved derived feature(s)", len(report.profiles))
    return _write(store, layout, report)


def _problem_type(
    store: ObjectStore, layout: ExperimentLayout, definition: ExperimentDefinition
) -> ProblemType:
    """What leakage screening needs. The authoritative type is resolved again at training."""
    if definition.requested_problem_type is not RequestedProblemType.AUTO:
        return ProblemType(definition.requested_problem_type.value)
    eda = read_model_if_exists(store, layout.eda, EdaReport)
    if eda is not None and eda.target.inferred_problem_type is not None:
        return eda.target.inferred_problem_type
    raise JobError(
        "PROBLEM_TYPE_UNKNOWN",
        "The problem type could not be inferred from the EDA, so derived features cannot be "
        "screened for leakage. Choose a problem type explicitly.",
    )


def _empty(definition: ExperimentDefinition) -> DerivedProfileReport:
    return DerivedProfileReport(
        experiment_id=definition.experiment_id,
        generated_at=datetime.now(UTC),
        target_column=definition.target_column,
        profiles=[],
    )


def _write(
    store: ObjectStore, layout: ExperimentLayout, report: DerivedProfileReport
) -> DerivedProfileReport:
    write_model(store, layout.derived_profiles, report)
    LOGGER.info("Wrote %s", layout.derived_profiles)
    return report


def _handler(args: argparse.Namespace, store: ObjectStore, layout: ExperimentLayout) -> None:
    run_derived_profiling(store, layout, experiment_id=args.experiment_id)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("ML Factory derived-feature profiling job")
    parser.prog = "derived-profiling-job"
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_entrypoint(
        build_parser(),
        _handler,
        argv,
        failure_uri=lambda _args, layout: layout.path(f"{FAILURE_DIR}/failure.json"),
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
