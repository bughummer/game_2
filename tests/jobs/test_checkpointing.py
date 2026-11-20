"""The training-stage checkpoint module is inert (stock-image migration, requirement 5.2, 5.4).

Spot checkpoint/resume is dropped: ``RESUMABLE_MODELS`` is empty and ``checkpointing_factory``
returns the plain plugin unchanged (no read, no write). The module and its public symbol are
kept so the training job's import resolves. These tests assert the no-op behaviour and that a
training run still produces a model without writing any checkpoint.
"""

import numpy as np
import pandas as pd
import pytest

from jobs.training.checkpointing import (
    RESUMABLE_MODELS,
    checkpointing_factory,
    is_resumable,
)
from jobs.training.main import run_training
from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.model import ModelMetadata
from ml_engine.io import read_model, write_model
from ml_engine.models import TrainingContext, get_plugin
from tests.support.experiments import make_config


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(300, 5)), columns=list("abcde"))
    y = pd.Series((x["a"] + rng.normal(size=300) > 0).astype(int).astype(str))
    context = TrainingContext(
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        class_labels=["0", "1"],
        class_counts={k: int(v) for k, v in y.value_counts().items()},
    )
    return x, y, context


def _fit(plugin, x, y, context):
    params = get_plugin(plugin.name).resolve_params({}, context)
    return plugin.fit(plugin.build(params, context), x, y, context)


def test_the_resumable_model_set_is_empty():
    """Decision B4: no model resumes from a Spot checkpoint."""
    assert frozenset() == RESUMABLE_MODELS


@pytest.mark.parametrize("model", ["xgboost", "logistic_regression", "random_forest"])
def test_the_factory_returns_the_plain_plugin_unwrapped(model, tmp_path):
    """Every model is restart-on-interruption: the factory wraps nothing (Req 5.2)."""
    assert not is_resumable(model)
    assert checkpointing_factory(tmp_path)(model) is get_plugin(model)


def test_the_factory_writes_no_checkpoint_files(xy, tmp_path):
    """A would-be-resumable model fitted through the factory writes nothing to the directory."""
    x, y, context = xy
    _fit(checkpointing_factory(tmp_path)("xgboost"), x, y, context)
    assert list(tmp_path.iterdir()) == []


def test_without_a_checkpoint_the_model_is_identical(xy, tmp_path):
    """The no-op factory produces exactly the plain plugin's model."""
    x, y, context = xy
    plain = _fit(get_plugin("xgboost"), x, y, context).predict_proba(x)
    wrapped = _fit(checkpointing_factory(tmp_path)("xgboost"), x, y, context).predict_proba(x)
    np.testing.assert_array_equal(plain, wrapped)


def test_training_cli_still_accepts_a_checkpoint_dir_but_writes_nothing(
    store, layout, dataset_csv, tmp_path, monkeypatch
):
    """The --checkpoint-dir flag is kept for signature compatibility; it is now inert."""
    from jobs.preprocessing.main import run_preparation
    from jobs.profiling.main import run_profiling
    from jobs.training.main import main as training_main

    monkeypatch.setattr("jobs._common.runtime.build_store", lambda: store)
    run_profiling(
        store, layout, experiment_id="exp-test", dataset_uri=dataset_csv, target_column="churned"
    )
    write_model(
        store, layout.experiment_config, make_config(layout, "exp-test", dataset_csv, ["xgboost"])
    )
    run_preparation(store, layout, experiment_id="exp-test")

    code = training_main(
        [
            "--experiment-id",
            "exp-test",
            "--artifact-base",
            layout.base,
            "--model-name",
            "xgboost",
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
        ]
    )

    assert code == 0
    # Inert: no checkpoint files are written even when a directory is given.
    assert not (tmp_path / "ckpt").exists() or not any((tmp_path / "ckpt").iterdir())
    assert (
        read_model(store, layout.model_metadata("xgboost"), ModelMetadata).primary_score is not None
    )


def test_checkpointing_does_not_change_recorded_hyperparameters(
    store, layout, dataset_csv, tmp_path
):
    from jobs.preprocessing.main import run_preparation
    from jobs.profiling.main import run_profiling

    run_profiling(
        store, layout, experiment_id="exp-test", dataset_uri=dataset_csv, target_column="churned"
    )
    write_model(
        store, layout.experiment_config, make_config(layout, "exp-test", dataset_csv, ["xgboost"])
    )
    run_preparation(store, layout, experiment_id="exp-test")

    plain = run_training(store, layout, experiment_id="exp-test", model_name="xgboost")
    wrapped = run_training(
        store,
        layout,
        experiment_id="exp-test",
        model_name="xgboost",
        plugin_factory=checkpointing_factory(tmp_path),
    )
    assert plain.hyperparameters == wrapped.hyperparameters
    assert plain.primary_score == wrapped.primary_score
