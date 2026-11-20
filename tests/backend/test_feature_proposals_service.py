"""Derived-feature approval, profiling and use in training.

Pins the three things that must hold end to end:
* approving never loads the dataset in the control plane — profiling is a SageMaker job;
* approved features appear in feature review and can be kept or excluded like any column;
* what is approved (and not excluded) is exactly what is frozen into the training config.
Also: concurrent "start training" calls launch one run, not two.
"""

import threading
from datetime import UTC, datetime

import pandas as pd
import pytest

from backend.errors import ConflictError, FeatureDisabledError, ValidationError
from backend.orchestration.base import ExecutionHandle
from backend.repositories import ExperimentRepository
from backend.schemas import ApproveProposalsRequest, ProposeFeaturesRequest, TrainingRequest
from backend.services import ExperimentService, FeatureProposalService
from jobs.derived_profiling.main import run_derived_profiling
from jobs.profiling.main import run_profiling
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.experiment import ExperimentDefinition
from ml_engine.contracts.proposals import (
    DerivedProfileReport,
    DifferenceProposal,
    FeatureProposalReport,
    RatioProposal,
)
from ml_engine.io import ExperimentLayout, join_uri, read_model, write_model
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

EXP = "exp-derived01"
NOW = datetime(2026, 10, 14, tzinfo=UTC)
RATIO = "charges_per_tenure_month"
GAP = "score_gap"


class SpyStore(InMemoryObjectStore):
    """Records every full-object download, which is how a dataset gets loaded."""

    def __init__(self) -> None:
        super().__init__()
        self.downloads: list[str] = []

    def download(self, uri, destination):
        self.downloads.append(uri)
        return super().download(uri, destination)


class RecordingOrchestrator:
    """Records what was started; optionally runs it with the real stages (InlineOrchestrator)."""

    name = "recording"

    def __init__(self, store, *, run=False, fail=False, gate: threading.Event | None = None):
        self._inline = InlineOrchestrator(store) if run else None
        self._fail = fail
        self._gate = gate
        self.started: list[tuple[str, str]] = []

    def start_eda(self, record):
        raise AssertionError("not used")

    def start_training(self, record):
        self.started.append(("training", record.experiment_id))
        if self._gate is not None:
            self._gate.wait(5)
        return ExecutionHandle(execution_id=f"{record.experiment_id}.training.t{len(self.started)}")

    def start_derived_profiling(self, record):
        if self._fail:
            raise RuntimeError("AccessDenied")
        self.started.append(("derived_profiling", record.experiment_id))
        if self._inline is not None:
            return self._inline.start_derived_profiling(record)
        return ExecutionHandle(execution_id=f"{record.experiment_id}.derivedprofiling.abc")

    def describe(self, execution_id):
        return {}


@pytest.fixture
def store():
    return SpyStore()


@pytest.fixture
def layout():
    return ExperimentLayout.for_experiment(ARTIFACT_ROOT, EXP)


@pytest.fixture
def dataset_uri(store, approved_prefix) -> str:
    rows = 200
    frame = pd.DataFrame(
        {
            "monthly_charges": [40.0 + (i % 50) for i in range(rows)],
            "tenure_months": [float(1 + i % 24) for i in range(rows)],
            "score_after": [(i % 5) + (i % 2) for i in range(rows)],
            "score_before": [i % 5 for i in range(rows)],
            "churned": [i % 2 for i in range(rows)],
        }
    )
    uri = join_uri(approved_prefix, "derived.csv")
    store.write_bytes(uri, frame.to_csv(index=False).encode())
    return uri


@pytest.fixture
def repository(settings, store):
    return ExperimentRepository(settings=settings, store=store)


@pytest.fixture
def experiment(repository, store, layout, dataset_uri):
    repository.create(
        ExperimentDefinition(
            experiment_id=EXP,
            name="derived",
            created_by="dev@local",
            created_at=NOW,
            dataset={"uri": dataset_uri, "file_format": "csv"},
            target_column="churned",
            artifact_prefix=layout.base,
        )
    )
    run_profiling(
        store,
        layout,
        experiment_id=EXP,
        dataset_uri=dataset_uri,
        target_column="churned",
        file_format="csv",
    )
    write_model(
        store,
        layout.feature_proposals,
        FeatureProposalReport(
            experiment_id=EXP,
            generated_at=NOW,
            candidates=[
                RatioProposal(
                    name=RATIO,
                    numerator="monthly_charges",
                    denominator="tenure_months",
                    rationale="spend per month of tenure",
                    available_at_prediction_time=True,
                ),
                DifferenceProposal(
                    name=GAP,
                    left="score_after",
                    right="score_before",
                    rationale="change in score",
                    available_at_prediction_time=True,
                ),
            ],
        ),
    )
    store.downloads.clear()  # only count what happens after setup
    return EXP


def _proposals(settings, repository, store, orchestrator):
    return FeatureProposalService(
        settings=settings,
        repository=repository,
        store=store,
        client=None,
        orchestrator=orchestrator,
    )


def _experiments(settings, repository, store, orchestrator):
    return ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )


# --- approval never loads the dataset in the control plane ------------------------------------


def test_approving_starts_a_profiling_job_and_loads_no_data(
    settings, repository, store, layout, experiment, dataset_uri
):
    orchestrator = RecordingOrchestrator(store)
    service = _proposals(settings, repository, store, orchestrator)

    result = service.approve(EXP, ApproveProposalsRequest(approved_names=[RATIO]))

    assert result.report.approved_names == [RATIO]
    assert result.profiling_execution_id == f"{EXP}.derivedprofiling.abc"
    assert orchestrator.started == [("derived_profiling", EXP)]
    assert dataset_uri not in store.downloads
    # Earlier profiles are cleared until the job writes fresh ones.
    assert read_model(store, layout.derived_profiles, DerivedProfileReport).profiles == []


def test_approving_none_starts_no_job(settings, repository, store, layout, experiment):
    orchestrator = RecordingOrchestrator(store)
    service = _proposals(settings, repository, store, orchestrator)

    result = service.approve(EXP, ApproveProposalsRequest(approved_names=[]))

    assert result.profiling_execution_id is None
    assert orchestrator.started == []
    assert read_model(store, layout.derived_profiles, DerivedProfileReport).profiles == []


def test_a_failed_profiling_start_still_records_the_approval(
    settings, repository, store, layout, experiment
):
    service = _proposals(settings, repository, store, RecordingOrchestrator(store, fail=True))

    result = service.approve(EXP, ApproveProposalsRequest(approved_names=[RATIO]))

    assert result.profiling_execution_id is None
    saved = read_model(store, layout.feature_proposals, FeatureProposalReport)
    assert saved.approved_names == [RATIO]


def test_only_offered_proposals_can_be_approved(settings, repository, store, experiment):
    service = _proposals(settings, repository, store, RecordingOrchestrator(store))

    with pytest.raises(ValidationError):
        service.approve(EXP, ApproveProposalsRequest(approved_names=["invented"]))


def test_proposing_needs_bedrock(settings, repository, store, experiment):
    service = _proposals(settings, repository, store, RecordingOrchestrator(store))

    with pytest.raises(FeatureDisabledError):
        service.generate(EXP, ProposeFeaturesRequest())


# --- the profiling job ------------------------------------------------------------------------


def test_the_profiling_job_profiles_exactly_the_approved_features(
    settings, repository, store, layout, experiment
):
    service = _proposals(settings, repository, store, RecordingOrchestrator(store, run=True))

    service.approve(EXP, ApproveProposalsRequest(approved_names=[RATIO, GAP]))

    profiles = read_model(store, layout.derived_profiles, DerivedProfileReport)
    assert {p.name for p in profiles.profiles} == {RATIO, GAP}
    assert all(p.unique_count > 0 for p in profiles.profiles)


def test_the_profiling_job_with_nothing_approved_writes_an_empty_report(store, layout, experiment):
    report = run_derived_profiling(store, layout, experiment_id=EXP)

    assert report.profiles == []


# --- feature review and training --------------------------------------------------------------


def test_approved_features_show_in_review_pending_then_profiled(
    settings, repository, store, experiment
):
    pending = _proposals(settings, repository, store, RecordingOrchestrator(store))
    pending.approve(EXP, ApproveProposalsRequest(approved_names=[RATIO]))
    experiments = _experiments(settings, repository, store, RecordingOrchestrator(store))

    row = next(f for f in experiments.feature_review(EXP).features if f.feature == RATIO)
    assert (row.origin, row.status, row.selected, row.leakage_risk) == (
        "derived",
        "approved",
        True,
        "pending",
    )

    _proposals(settings, repository, store, RecordingOrchestrator(store, run=True)).approve(
        EXP, ApproveProposalsRequest(approved_names=[RATIO])
    )
    row = next(f for f in experiments.feature_review(EXP).features if f.feature == RATIO)
    assert row.leakage_risk != "pending" and row.dtype != "pending"


def test_training_freezes_approved_features_minus_the_excluded_ones(
    settings, repository, store, layout, experiment
):
    _proposals(settings, repository, store, RecordingOrchestrator(store)).approve(
        EXP, ApproveProposalsRequest(approved_names=[RATIO, GAP])
    )
    experiments = _experiments(settings, repository, store, RecordingOrchestrator(store))
    experiments.save_features(
        EXP, selected_features=["monthly_charges", "tenure_months", RATIO]
    )  # GAP excluded

    experiments.start_training(EXP, TrainingRequest(models=["logistic_regression"]))

    config = read_model(store, layout.experiment_config, ExperimentConfig)
    assert [p.name for p in config.derived_features] == [RATIO]
    assert GAP in config.feature_selection.excluded_features


# --- one start at a time ----------------------------------------------------------------------


def test_concurrent_training_starts_launch_one_run(settings, repository, store, layout, experiment):
    gate = threading.Event()
    orchestrator = RecordingOrchestrator(store, gate=gate)
    experiments = _experiments(settings, repository, store, orchestrator)
    experiments.save_features(EXP, selected_features=["monthly_charges", "tenure_months"])
    request = TrainingRequest(models=["logistic_regression"])
    errors: list[Exception] = []

    def start():
        try:
            experiments.start_training(EXP, request)
        except ConflictError as error:
            errors.append(error)

    first = threading.Thread(target=start)
    first.start()
    while not orchestrator.started:  # first call is inside the orchestrator, holding the lock
        threading.Event().wait(0.01)
    start()  # the concurrent second call
    gate.set()
    first.join(5)

    assert orchestrator.started == [("training", EXP)]
    assert len(errors) == 1


# --- user-defined features --------------------------------------------------------------------


def _user_ratio(name="charges_per_month", **overrides):
    spec = {
        "name": name,
        "numerator": "monthly_charges",
        "denominator": "tenure_months",
        "rationale": "defined by the analyst",
        "available_at_prediction_time": True,
        **overrides,
    }
    return RatioProposal(**spec)


def test_a_user_defined_feature_is_added_approved_and_profiled_without_bedrock(
    settings, repository, store, layout, experiment
):
    orchestrator = RecordingOrchestrator(store)
    service = _proposals(settings, repository, store, orchestrator)
    assert not service.enabled  # Bedrock is off

    result = service.define(EXP, [_user_ratio()])

    assert result.added == ["charges_per_month"]
    assert result.rejected == []
    assert result.report.approved_names == ["charges_per_month"]
    assert orchestrator.started == [("derived_profiling", EXP)]
    saved = read_model(store, layout.feature_proposals, FeatureProposalReport)
    assert "charges_per_month" in {c.name for c in saved.candidates}


def test_user_defined_features_keep_earlier_approvals(settings, repository, store, experiment):
    service = _proposals(settings, repository, store, RecordingOrchestrator(store))
    service.approve(EXP, ApproveProposalsRequest(approved_names=[RATIO]))

    result = service.define(EXP, [_user_ratio()])

    assert result.report.approved_names == [RATIO, "charges_per_month"]


def test_a_user_defined_feature_gets_no_shortcut_past_validation(
    settings, repository, store, experiment
):
    service = _proposals(settings, repository, store, RecordingOrchestrator(store))

    result = service.define(
        EXP,
        [
            _user_ratio(name="leak", numerator="churned"),  # reads the target
            _user_ratio(name="ghost", denominator="no_such_column"),
            _user_ratio(name=RATIO),  # name already taken by an earlier candidate
            _user_ratio(name="ok_feature"),
        ],
    )

    reasons = {r.name: r.reason.value for r in result.rejected}
    assert reasons == {
        "leak": "target_reference",
        "ghost": "unknown_column",
        RATIO: "already_proposed",
    }
    assert result.added == ["ok_feature"]


def test_defining_without_approval_leaves_it_as_a_candidate(
    settings, repository, store, experiment
):
    orchestrator = RecordingOrchestrator(store)
    service = _proposals(settings, repository, store, orchestrator)

    result = service.define(EXP, [_user_ratio()], approve=False)

    assert result.added == ["charges_per_month"]
    assert result.report.approved_names == []
    assert orchestrator.started == []


def test_a_user_defined_feature_reaches_training(settings, repository, store, layout, experiment):
    _proposals(settings, repository, store, RecordingOrchestrator(store)).define(
        EXP, [_user_ratio()]
    )
    experiments = _experiments(settings, repository, store, RecordingOrchestrator(store))
    experiments.save_features(
        EXP, selected_features=["monthly_charges", "tenure_months", "charges_per_month"]
    )

    experiments.start_training(EXP, TrainingRequest(models=["logistic_regression"]))

    config = read_model(store, layout.experiment_config, ExperimentConfig)
    assert [p.name for p in config.derived_features] == ["charges_per_month"]


def test_defining_needs_eda(settings, repository, store, layout, dataset_uri):
    repository.create(
        ExperimentDefinition(
            experiment_id="exp-noeda",
            name="no eda",
            created_at=NOW,
            dataset={"uri": dataset_uri, "file_format": "csv"},
            target_column="churned",
            artifact_prefix=ExperimentLayout.for_experiment(ARTIFACT_ROOT, "exp-noeda").base,
        )
    )
    service = _proposals(settings, repository, store, RecordingOrchestrator(store))

    with pytest.raises(ConflictError):
        service.define("exp-noeda", [_user_ratio()])


# --- training options -------------------------------------------------------------------------


def test_class_weighting_and_model_parameters_are_frozen_into_the_config(
    settings, repository, store, layout, experiment
):
    from ml_engine.contracts.common import ClassWeighting

    experiments = _experiments(settings, repository, store, RecordingOrchestrator(store))
    experiments.save_features(EXP, selected_features=["monthly_charges", "tenure_months"])

    experiments.start_training(
        EXP,
        TrainingRequest(
            models=["xgboost"],
            class_weighting=ClassWeighting.NONE,
            model_parameters={"xgboost": {"scale_pos_weight": 3}},
        ),
    )

    config = read_model(store, layout.experiment_config, ExperimentConfig)
    assert config.class_weighting is ClassWeighting.NONE
    assert config.models[0].params == {"scale_pos_weight": 3}


def test_parameters_for_a_model_not_in_the_run_are_refused(settings, repository, store, experiment):
    orchestrator = RecordingOrchestrator(store)
    experiments = _experiments(settings, repository, store, orchestrator)
    experiments.save_features(EXP, selected_features=["monthly_charges", "tenure_months"])

    with pytest.raises(ValidationError) as error:
        experiments.start_training(
            EXP,
            TrainingRequest(
                models=["logistic_regression"],
                model_parameters={"xgboost": {"scale_pos_weight": 3}},
            ),
        )

    assert error.value.details["models_with_parameters"] == ["xgboost"]
    assert orchestrator.started == []
