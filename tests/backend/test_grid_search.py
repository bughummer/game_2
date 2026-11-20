"""Grid search: expansion, validation, guards and a real multi-point run (grid-search spec).

Offline, with the in-memory store and the inline orchestrator. A grid point is an ordinary
ModelSpec, so these tests assert the expansion/validation at the service layer and that every point
becomes a distinct, loadable model version through the real stages.
"""

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from backend.config import Settings
from backend.errors import ValidationError
from backend.repositories import ExperimentRepository
from backend.schemas import TrainingRequest
from backend.services import ExperimentService
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.grid import ParameterGrid
from ml_engine.io import ExperimentLayout, read_model
from ml_engine.models import base_model_name, get_plugin, grid_point_id
from ml_engine.models.grid import expand_grid
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.golden import golden_frame
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

FEATURES = ["tenure_months", "monthly_charges", "contract", "is_business", "signup_date"]
TARGET = "churned"


# --- registry: grid-point ids and base-plugin lookup ----------------------------------------


def test_get_plugin_resolves_a_grid_point_name_to_its_base():
    point = grid_point_id("xgboost", {"max_depth": 6})
    assert point.startswith("xgboost__")
    assert base_model_name(point) == "xgboost"
    assert get_plugin(point) is get_plugin("xgboost")


def test_grid_point_ids_are_stable_and_distinct():
    a = grid_point_id("xgboost", {"max_depth": 4, "reg_alpha": 0})
    b = grid_point_id("xgboost", {"reg_alpha": 0, "max_depth": 4})  # order-independent
    c = grid_point_id("xgboost", {"max_depth": 6, "reg_alpha": 0})
    assert a == b
    assert a != c


# --- pure expansion -------------------------------------------------------------------------


def test_expand_grid_is_the_cross_product():
    grid = ParameterGrid(grid={"xgboost": {"max_depth": [4, 6], "reg_alpha": [0, 0.1]}})
    points = expand_grid(grid, ["xgboost"])["xgboost"]
    combos = {tuple(sorted(p.params.items())) for p in points}
    assert len(points) == 4
    assert combos == {
        (("max_depth", 4), ("reg_alpha", 0)),
        (("max_depth", 4), ("reg_alpha", 0.1)),
        (("max_depth", 6), ("reg_alpha", 0)),
        (("max_depth", 6), ("reg_alpha", 0.1)),
    }


def test_a_model_without_a_grid_entry_yields_no_points():
    grid = ParameterGrid(grid={"xgboost": {"max_depth": [4]}})
    expanded = expand_grid(grid, ["xgboost", "logistic_regression"])
    assert len(expanded["xgboost"]) == 1
    assert expanded["logistic_regression"] == []


# --- parameter space validation -------------------------------------------------------------


def test_each_plugin_exposes_a_searchable_space():
    for name in ("xgboost", "logistic_regression", "random_forest", "elastic_net"):
        assert get_plugin(name).parameter_space()


def test_an_unknown_parameter_is_rejected():
    reasons = get_plugin("xgboost").validate_grid_value("not_a_param", 1)
    assert reasons and "searchable" in reasons[0]


def test_an_out_of_bound_value_is_rejected():
    reasons = get_plugin("xgboost").validate_grid_value("learning_rate", 5.0)
    assert reasons and "<=" in reasons[0]


# --- service-layer expansion and guards -----------------------------------------------------


def _service(store, **overrides):
    settings = Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn="arn:aws:iam::123456789012:role/mlf-sagemaker-job",
        **overrides,
    )
    repository = ExperimentRepository(settings=settings, store=store)
    orchestrator = InlineOrchestrator(store)
    experiments = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    return experiments, orchestrator


def _ready(experiments, store, name="churn.csv"):
    uri = write_dataset(store, name, golden_frame())
    record = experiments.create(name="g", dataset_uri=uri, target_column=TARGET)
    experiments.save_features(record.experiment_id, selected_features=FEATURES)
    return record


def test_a_grid_varying_a_non_parameter_key_is_rejected():
    store = InMemoryObjectStore()
    experiments, _ = _service(store)
    record = _ready(experiments, store)
    with pytest.raises(ValidationError) as excinfo:
        experiments.start_training(
            record.experiment_id,
            TrainingRequest(models=["xgboost"], parameter_grid={"features": {"x": [1]}}),
        )
    assert "model parameters only" in str(excinfo.value)


def test_a_grid_over_the_fit_ceiling_is_refused_and_submits_no_job():
    store = InMemoryObjectStore()
    # cv_max_fits small so a modest grid exceeds it under CV.
    experiments, orchestrator = _service(store, cv_max_fits=6)
    record = _ready(experiments, store)
    # 4 points x (K=3 + 1) = 16 fits > 6.
    with pytest.raises(ValidationError) as excinfo:
        experiments.start_training(
            record.experiment_id,
            TrainingRequest(
                models=["xgboost"],
                parameter_grid={"xgboost": {"max_depth": [3, 5], "reg_alpha": [0, 0.1]}},
                evaluation_mode="cv",
                cv_folds=3,
            ),
        )
    assert "exceed" in str(excinfo.value)
    layout = ExperimentLayout(base=record.artifact_prefix)
    assert not store.exists(layout.experiment_config)  # nothing frozen/submitted
    assert not orchestrator.failures


def test_a_grid_freezes_one_spec_per_point():
    store = InMemoryObjectStore()
    experiments, orchestrator = _service(store)
    record = _ready(experiments, store)
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(models=["xgboost"], parameter_grid={"xgboost": {"max_depth": [3, 5]}}),
    )
    assert not orchestrator.failures, orchestrator.failures
    layout = ExperimentLayout(base=record.artifact_prefix)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    specs = config.enabled_models_specs()
    assert len(specs) == 2
    assert all(base_model_name(s.name) == "xgboost" for s in specs)
    assert {s.params["max_depth"] for s in specs} == {3, 5}


def test_no_grid_behaves_exactly_as_today():
    store = InMemoryObjectStore()
    experiments, orchestrator = _service(store)
    record = _ready(experiments, store)
    experiments.start_training(record.experiment_id, TrainingRequest(models=["xgboost"]))
    assert not orchestrator.failures, orchestrator.failures
    layout = ExperimentLayout(base=record.artifact_prefix)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    specs = config.enabled_models_specs()
    assert [s.name for s in specs] == ["xgboost"]  # one spec, no '__' suffix


def test_every_grid_point_becomes_a_distinct_loadable_model_version():
    store = InMemoryObjectStore()
    experiments, orchestrator = _service(store)
    record = _ready(experiments, store)
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(models=["xgboost"], parameter_grid={"xgboost": {"max_depth": [3, 5]}}),
    )
    assert not orchestrator.failures, orchestrator.failures
    layout = ExperimentLayout(base=record.artifact_prefix)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    for spec in config.enabled_models_specs():
        assert store.exists(layout.model_artifact(spec.name)), spec.name
        assert store.exists(layout.model_metadata(spec.name)), spec.name
    # Ranked in the comparison.
    comparison = experiments.comparison(record.experiment_id)
    ranked = {e.model_name for e in comparison.models if e.rank is not None}
    assert ranked == {s.name for s in config.enabled_models_specs()}


# --- Property 1/2: expansion is the cross-product; fit count is points x (K+1) / x1 -----------

_param_values = st.lists(
    st.integers(min_value=1, max_value=20), min_size=1, max_size=4, unique=True
)


# Feature: grid-search, Property 1: grid expansion equals the per-parameter cross-product
@hyp_settings(max_examples=100, deadline=None, derandomize=True)
@given(
    max_depth=_param_values,
    min_child=_param_values,
)
def test_property_expansion_is_the_cross_product(max_depth, min_child):
    grid = ParameterGrid(grid={"xgboost": {"max_depth": max_depth, "min_child_weight": min_child}})
    points = expand_grid(grid, ["xgboost"])["xgboost"]
    assert len(points) == len(max_depth) * len(min_child)
    combos = {(p.params["max_depth"], p.params["min_child_weight"]) for p in points}
    assert combos == {(d, m) for d in max_depth for m in min_child}
    # Point ids are unique per combination.
    assert len({p.point_id for p in points}) == len(points)


# --- Property: a grid point's fit cost flows through the fold guard --------------------------


def test_cv_grid_fit_count_is_points_times_k_plus_one():
    store = InMemoryObjectStore()
    experiments, orchestrator = _service(store, cv_max_fits=60)
    record = _ready(experiments, store)
    # 4 points x (K=3 + 1) = 16 fits, under the ceiling.
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(
            models=["xgboost"],
            parameter_grid={"xgboost": {"max_depth": [3, 5], "min_child_weight": [1, 2]}},
            evaluation_mode="cv",
            cv_folds=3,
        ),
    )
    assert not orchestrator.failures, orchestrator.failures
    layout = ExperimentLayout(base=record.artifact_prefix)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    assert len(config.enabled_models_specs()) == 4
    assert config.split.cv is not None and config.split.cv.folds == 3


def test_a_grid_point_failure_leaves_the_others_intact(monkeypatch):
    store = InMemoryObjectStore()
    experiments, _orchestrator = _service(store)
    record = _ready(experiments, store)

    # Make exactly one point's training raise; the inline orchestrator records it as a failure
    # and continues to the others (per-point isolation, Req 5.3).
    from jobs.training import main as training_main

    real_run_training = training_main.run_training
    failed = {"name": None}

    def flaky(store_, layout_, *, experiment_id, model_name, **kw):
        if failed["name"] is None and model_name.startswith("xgboost__"):
            failed["name"] = model_name
            raise RuntimeError("boom")
        return real_run_training(
            store_, layout_, experiment_id=experiment_id, model_name=model_name, **kw
        )

    monkeypatch.setattr(training_main, "run_training", flaky)
    # InlineOrchestrator imported run_training by name; patch there too.
    from tests.support import inline_orchestrator

    monkeypatch.setattr(inline_orchestrator, "run_training", flaky)

    experiments.start_training(
        record.experiment_id,
        TrainingRequest(models=["xgboost"], parameter_grid={"xgboost": {"max_depth": [3, 5, 7]}}),
    )
    layout = ExperimentLayout(base=record.artifact_prefix)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    names = [s.name for s in config.enabled_models_specs()]
    # The failed point has a failure record; the others have metadata.
    assert store.exists(layout.model_failure(failed["name"]))
    survivors = [n for n in names if n != failed["name"]]
    assert survivors
    for name in survivors:
        assert store.exists(layout.model_metadata(name)), name
