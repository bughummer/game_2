# Feature: cross-validation, Property 13: the fold-count guard accepts a CV request iff
# 2 <= K <= cv_max_folds and models x (K + 1) <= cv_max_fits, and every rejection names the
# allowed range or the requested/allowed fold-fit counts.
"""Property 13 (fold-count guard) and unit checks for ``resolve_cv_folds`` (Req 1.4, 1.5).

The guard is a pure function, so the property calls it directly with no store, no AWS and no
experiment. Validates Requirements 1.4 (K range) and 1.5 (fit ceiling).
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.errors import ValidationError
from backend.orchestration.guards import resolve_cv_folds

PBT = settings(max_examples=100, deadline=None, derandomize=True)


@given(
    requested=st.one_of(st.none(), st.integers(min_value=-3, max_value=30)),
    model_count=st.integers(min_value=1, max_value=20),
    default_folds=st.integers(min_value=2, max_value=10),
    max_folds=st.integers(min_value=2, max_value=15),
    max_fits=st.integers(min_value=3, max_value=120),
)
@PBT
def test_fold_count_guard(requested, model_count, default_folds, max_folds, max_fits):
    # Keep the settings internally consistent, as the Settings validator guarantees.
    default_folds = min(default_folds, max_folds)
    resolved_k = default_folds if requested is None else requested
    accepts = 2 <= resolved_k <= max_folds and model_count * (resolved_k + 1) <= max_fits

    if accepts:
        folds = resolve_cv_folds(
            requested_folds=requested,
            evaluation_mode="cv",
            model_count=model_count,
            default_folds=default_folds,
            max_folds=max_folds,
            max_fits=max_fits,
        )
        assert folds == resolved_k
        return

    with pytest.raises(ValidationError) as error:
        resolve_cv_folds(
            requested_folds=requested,
            evaluation_mode="cv",
            model_count=model_count,
            default_folds=default_folds,
            max_folds=max_folds,
            max_fits=max_fits,
        )
    details = error.value.details
    if not (2 <= resolved_k <= max_folds):
        # Out of range: the rejection names the allowed range.
        assert details["reason"] == "CV_FOLDS_OUT_OF_RANGE"
        assert details["allowed_min"] == 2
        assert details["allowed_max"] == max_folds
        assert details["requested"] == resolved_k
    else:
        # In range but too many fits: the rejection names requested and allowed fits.
        assert details["reason"] == "CV_FIT_LIMIT_EXCEEDED"
        assert details["requested_fits"] == model_count * (resolved_k + 1)
        assert details["allowed_fits"] == max_fits


@given(requested=st.integers(min_value=2, max_value=10))
@PBT
def test_cv_folds_without_cv_mode_always_rejected(requested):
    for mode in (None, "single_split"):
        with pytest.raises(ValidationError) as error:
            resolve_cv_folds(
                requested_folds=requested,
                evaluation_mode=mode,
                model_count=1,
                default_folds=5,
                max_folds=10,
                max_fits=60,
            )
        assert error.value.details["reason"] == "CV_FOLDS_WITHOUT_CV_MODE"


def test_single_split_never_applies_the_fit_ceiling():
    # 20 models would exceed any CV ceiling, but single-split mode resolves without a check.
    assert (
        resolve_cv_folds(
            requested_folds=None,
            evaluation_mode=None,
            model_count=20,
            default_folds=5,
            max_folds=10,
            max_fits=60,
        )
        == 5
    )
