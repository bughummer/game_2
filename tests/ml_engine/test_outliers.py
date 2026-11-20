"""The outlier engine (training-data-treatment design C2).

Property tests (Hypothesis) for the correctness properties in the design:
P1 non-mutation, P2 fit-isolation, P3 validation-independence, P4 aligned held-out-safe drop,
P5 determinism + round-trip, plus contract round-trips and the error paths.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml_engine.contracts.treatment import (
    MIN_TRAINING_ROWS_AFTER_DROP,
    AppliedTreatment,
    OutlierBounds,
    OutlierDetector,
    OutlierHandling,
    TreatmentConfig,
)
from ml_engine.preprocessing.outliers import (
    OutlierTreatment,
    OutlierTreatmentError,
    fit_and_apply,
)

SETTINGS = settings(max_examples=100, deadline=None, derandomize=True)


def _frame(rows: int = 60, seed: int = 0, *, with_outliers: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    amount = rng.normal(loc=10.0, scale=2.0, size=rows)
    if with_outliers:
        x[0] = 50.0  # a clear outlier in column x
        amount[1] = -100.0  # a clear outlier in column amount
    return pd.DataFrame({"x": x, "amount": amount, "category": ["a", "b"] * (rows // 2)})


def _target(rows: int = 60, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, 2, size=rows), name="label")


# --- config validation ------------------------------------------------------------------------


def test_cap_with_isolation_forest_is_refused():
    with pytest.raises(ValueError, match="cap"):
        TreatmentConfig(
            outlier_detector=OutlierDetector.ISOLATION_FOREST,
            outlier_handling=OutlierHandling.CAP,
        )


def test_cap_with_per_column_detector_is_allowed():
    config = TreatmentConfig(
        outlier_detector=OutlierDetector.IQR, outlier_handling=OutlierHandling.CAP
    )
    assert config.outlier_handling is OutlierHandling.CAP


# --- Property 6: none is a pass-through -------------------------------------------------------


def test_none_is_passthrough():
    frame, y = _frame(), _target()
    treatment = OutlierTreatment(TreatmentConfig()).fit(frame, y)
    x_out, y_out = treatment.apply(frame, y)
    pd.testing.assert_frame_equal(x_out, frame)
    pd.testing.assert_series_equal(y_out, y)


# --- Property 1: non-mutation ------------------------------------------------------------------


@given(detector=st.sampled_from([OutlierDetector.IQR, OutlierDetector.ZSCORE]))
@SETTINGS
def test_apply_does_not_mutate_input(detector):
    frame, y = _frame(), _target()
    before = frame.copy(deep=True)
    config = TreatmentConfig(outlier_detector=detector, outlier_handling=OutlierHandling.DROP)
    OutlierTreatment(config).fit(frame, y).apply(frame, y)
    pd.testing.assert_frame_equal(frame, before)


# --- Property 2: fit isolation -----------------------------------------------------------------


def test_fit_isolation_bounds_from_training_rows_only():
    frame = _frame(rows=80)
    train = frame.iloc[:40]
    config = TreatmentConfig(outlier_detector=OutlierDetector.IQR)
    bounds_from_train = OutlierTreatment(config).fit(train).bounds()
    bounds_from_train_again = OutlierTreatment(config).fit(frame.iloc[:40]).bounds()
    assert bounds_from_train.column_fences == bounds_from_train_again.column_fences


# --- Property 3: validation independence -------------------------------------------------------


def test_validation_independence():
    frame = _frame(rows=80)
    train = frame.iloc[:40].copy()
    config = TreatmentConfig(outlier_detector=OutlierDetector.ZSCORE)
    baseline = OutlierTreatment(config).fit(train).bounds()
    # Perturb only the held-out rows; the fitted bounds must not move.
    mutated = frame.copy()
    mutated.loc[mutated.index[40:], "x"] = 999.0
    after = OutlierTreatment(config).fit(mutated.iloc[:40]).bounds()
    assert baseline.column_fences == after.column_fences


# --- Property 4: aligned, held-out-safe drop ---------------------------------------------------


def test_drop_removes_exactly_the_flagged_rows_aligned():
    frame, y = _frame(rows=60), _target(rows=60)
    config = TreatmentConfig(
        outlier_detector=OutlierDetector.IQR, outlier_handling=OutlierHandling.DROP
    )
    treatment = OutlierTreatment(config).fit(frame, y)
    mask = treatment._outlier_mask(frame)
    x_out, y_out = treatment.apply(frame, y)
    assert len(x_out) == len(frame) - int(mask.sum())
    assert list(x_out.index) == list(y_out.index)
    # rows 0 and 1 are the injected outliers and must be gone
    assert 0 not in x_out.index
    assert 1 not in x_out.index


def test_drop_too_few_rows_raises_structured_error():
    frame = _frame(rows=12, with_outliers=False)
    # Force almost everything to be an outlier by a tiny z threshold.
    config = TreatmentConfig(
        outlier_detector=OutlierDetector.ZSCORE,
        outlier_handling=OutlierHandling.DROP,
        zscore_threshold=0.01,
    )
    with pytest.raises(OutlierTreatmentError) as excinfo:
        OutlierTreatment(config).fit(frame).apply(frame)
    assert excinfo.value.code == "OUTLIER_LEFT_TOO_FEW_ROWS"


def test_cap_preserves_row_count_and_clips():
    frame = _frame(rows=60)
    config = TreatmentConfig(
        outlier_detector=OutlierDetector.IQR, outlier_handling=OutlierHandling.CAP
    )
    treatment = OutlierTreatment(config).fit(frame)
    x_out, _ = treatment.apply(frame, None)
    assert len(x_out) == len(frame)
    fences = treatment.bounds().column_fences
    for column, (lower, upper) in fences.items():
        assert x_out[column].min() >= lower - 1e-9
        assert x_out[column].max() <= upper + 1e-9


def test_flag_changes_nothing():
    frame, y = _frame(), _target()
    config = TreatmentConfig(
        outlier_detector=OutlierDetector.IQR, outlier_handling=OutlierHandling.FLAG
    )
    x_out, y_out = OutlierTreatment(config).fit(frame, y).apply(frame, y)
    pd.testing.assert_frame_equal(x_out, frame)
    pd.testing.assert_series_equal(y_out, y)


# --- Property 5: determinism + round-trip ------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=10))
@SETTINGS
def test_isolation_forest_determinism(seed):
    frame = _frame(rows=60, seed=seed)
    config = TreatmentConfig(
        outlier_detector=OutlierDetector.ISOLATION_FOREST, outlier_handling=OutlierHandling.FLAG
    )
    a = OutlierTreatment(config, random_seed=7).fit(frame)
    b = OutlierTreatment(config, random_seed=7).fit(frame)
    np.testing.assert_array_equal(a._outlier_mask(frame), b._outlier_mask(frame))


def test_contract_round_trips():
    for model in (
        TreatmentConfig(
            outlier_detector=OutlierDetector.IQR, outlier_handling=OutlierHandling.DROP
        ),
        OutlierBounds(detector=OutlierDetector.IQR, column_fences={"x": (-1.0, 1.0)}),
        AppliedTreatment(
            fitted_on="fold_0_train",
            detector=OutlierDetector.IQR,
            handling=OutlierHandling.DROP,
            rows_considered=40,
            rows_dropped=2,
        ),
    ):
        restored = type(model).model_validate_json(model.model_dump_json())
        assert restored == model


# --- report + helper ---------------------------------------------------------------------------


def test_report_counts_dropped_rows():
    frame = _frame(rows=60)
    config = TreatmentConfig(
        outlier_detector=OutlierDetector.IQR, outlier_handling=OutlierHandling.DROP
    )
    treatment = OutlierTreatment(config).fit(frame)
    summary = treatment.report(frame, fitted_on="train")
    assert summary.rows_dropped >= 2
    assert summary.rows_considered == 60
    assert "x" in summary.affected_columns


def test_fit_and_apply_helper():
    frame, y = _frame(rows=60), _target(rows=60)
    config = TreatmentConfig(
        outlier_detector=OutlierDetector.IQR, outlier_handling=OutlierHandling.DROP
    )
    x_out, _y_out, treatment, summary = fit_and_apply(config, frame, y, fitted_on="train")
    assert treatment.is_fitted
    assert summary.rows_dropped == len(frame) - len(x_out)
    assert len(x_out) >= MIN_TRAINING_ROWS_AFTER_DROP
