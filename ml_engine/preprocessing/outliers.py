"""The pure outlier engine (training-data-treatment design C2).

``OutlierTreatment`` is a separate fitted object composed *around* ``FeaturePipeline``: it operates
on rows ``(X, y)`` before the column-wise pipeline fits, so it can drop rows or cap values — things
the ``ColumnTransformer`` cannot do. It is fitted on training rows only (per fold under CV), and
its fitted state (``OutlierBounds``) is enough to reproduce the treatment on the same rows.

Order in the chain: outlier handling (here) -> imputation -> scaling -> encoding -> model fit.
Validation and test rows are never outlier-removed; they run the pipeline with the training fold's
fitted state and every held-out row is scored.

Pure ``ml_engine``: no IO, no AWS. A ``none`` detector is a pass-through, so the chain is
byte-for-byte today's chain.
"""

import numpy as np
import pandas as pd

from ml_engine.contracts.treatment import (
    MIN_TRAINING_ROWS_AFTER_DROP,
    AppliedTreatment,
    OutlierBounds,
    OutlierDetector,
    OutlierHandling,
    TreatmentConfig,
)


class OutlierTreatmentError(RuntimeError):
    """A treatment cannot be applied (e.g. a ``drop`` would leave too few training rows)."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]


class OutlierTreatment:
    """Fit outlier bounds on training rows, then flag/cap/drop on a frame.

    Typical use in preparation::

        treatment = OutlierTreatment(config, random_seed=seed).fit(X_train, y_train)
        X_treated, y_treated = treatment.apply(X_train, y_train)

    Under CV each fold builds and fits its own ``OutlierTreatment`` on that fold's training rows.
    """

    def __init__(self, config: TreatmentConfig, *, random_seed: int = 42) -> None:
        self.config = config
        self.random_seed = random_seed
        self._fitted = False
        self._bounds = OutlierBounds(detector=config.outlier_detector)
        self._forest: object | None = None
        self._fit_columns: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def is_noop(self) -> bool:
        return self.config.is_noop

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series | None = None) -> "OutlierTreatment":  # noqa: ARG002 - y kept for a scikit-learn-style fit(X, y) signature
        """Learn bounds from training rows only. Deterministic for a given config + seed."""
        detector = self.config.outlier_detector
        if detector is OutlierDetector.NONE:
            self._fitted = True
            return self
        self._fit_columns = _numeric_columns(x_train)
        if detector in (OutlierDetector.IQR, OutlierDetector.ZSCORE):
            self._bounds = OutlierBounds(detector=detector, column_fences=self._fit_fences(x_train))
        elif detector is OutlierDetector.ISOLATION_FOREST:
            self._forest = self._fit_forest(x_train)
            self._bounds = OutlierBounds(
                detector=detector,
                forest_seed=self.random_seed,
                forest_contamination=self.config.isolation_contamination,
            )
        self._fitted = True
        return self

    def _fit_fences(self, x_train: pd.DataFrame) -> dict[str, tuple[float, float]]:
        fences: dict[str, tuple[float, float]] = {}
        for column in self._fit_columns:
            values = x_train[column].to_numpy(dtype="float64")
            values = values[~np.isnan(values)]
            if values.size == 0:
                continue
            if self.config.outlier_detector is OutlierDetector.IQR:
                q1, q3 = np.quantile(values, [0.25, 0.75])
                span = (q3 - q1) * self.config.iqr_multiplier
                lower, upper = q1 - span, q3 + span
            else:  # zscore
                mean = float(np.mean(values))
                std = float(np.std(values))
                if std == 0.0:
                    lower, upper = mean, mean
                else:
                    delta = self.config.zscore_threshold * std
                    lower, upper = mean - delta, mean + delta
            fences[column] = (float(lower), float(upper))
        return fences

    def _fit_forest(self, x_train: pd.DataFrame) -> object:
        from sklearn.ensemble import IsolationForest

        forest = IsolationForest(
            contamination=self.config.isolation_contamination,
            random_state=self.random_seed,
        )
        forest.fit(self._forest_matrix(x_train))
        return forest

    def _forest_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        # Column order is fixed by the training fit so prediction matches fit.
        matrix = frame.reindex(columns=self._fit_columns).to_numpy(dtype="float64")
        return np.nan_to_num(matrix, nan=0.0)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise OutlierTreatmentError(
                "OutlierTreatment must be fitted before it can be used.", code="NOT_FITTED"
            )

    def bounds(self) -> OutlierBounds:
        self._require_fitted()
        return self._bounds

    def _outlier_mask(self, x_frame: pd.DataFrame) -> np.ndarray:
        """Boolean mask over rows of ``x_frame``: True where the row is a detected outlier."""
        detector = self.config.outlier_detector
        if detector is OutlierDetector.NONE:
            return np.zeros(len(x_frame), dtype=bool)
        if detector in (OutlierDetector.IQR, OutlierDetector.ZSCORE):
            mask = np.zeros(len(x_frame), dtype=bool)
            for column, (lower, upper) in self._bounds.column_fences.items():
                if column not in x_frame.columns:
                    continue
                col = x_frame[column].to_numpy(dtype="float64")
                below = col < lower
                above = col > upper
                mask |= np.where(np.isnan(col), False, below | above)
            return mask
        # isolation_forest: predict returns -1 for anomalies.
        assert self._forest is not None
        predictions = self._forest.predict(self._forest_matrix(x_frame))  # type: ignore[attr-defined]
        return predictions == -1

    def _affected_columns(self, x_frame: pd.DataFrame) -> list[str]:
        if self.config.outlier_detector not in (OutlierDetector.IQR, OutlierDetector.ZSCORE):
            return list(self._fit_columns)
        affected: list[str] = []
        for column, (lower, upper) in self._bounds.column_fences.items():
            if column not in x_frame.columns:
                continue
            col = x_frame[column].to_numpy(dtype="float64")
            out = np.where(np.isnan(col), False, (col < lower) | (col > upper))
            if out.any():
                affected.append(column)
        return affected

    def apply(
        self, x_frame: pd.DataFrame, y: pd.Series | None = None
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Apply the fitted treatment to a frame. flag: unchanged; cap: winsorise; drop: remove."""
        self._require_fitted()
        if self.config.outlier_detector is OutlierDetector.NONE:
            return x_frame, y
        mask = self._outlier_mask(x_frame)
        handling = self.config.outlier_handling
        if handling is OutlierHandling.FLAG:
            return x_frame, y
        if handling is OutlierHandling.CAP:
            return self._cap(x_frame), y
        # drop: remove the detected rows from X and y, index-aligned.
        kept = ~mask
        remaining = int(kept.sum())
        if remaining < MIN_TRAINING_ROWS_AFTER_DROP:
            raise OutlierTreatmentError(
                f"outlier drop would leave {remaining} training row(s), below the minimum "
                f"{MIN_TRAINING_ROWS_AFTER_DROP}; {int(mask.sum())} of {len(x_frame)} flagged",
                code="OUTLIER_LEFT_TOO_FEW_ROWS",
            )
        kept_index = x_frame.index[kept]
        x_kept = x_frame.loc[kept_index]
        y_kept = None if y is None else y.loc[kept_index]
        return x_kept, y_kept

    def _cap(self, x_frame: pd.DataFrame) -> pd.DataFrame:
        capped = x_frame.copy()
        for column, (lower, upper) in self._bounds.column_fences.items():
            if column not in capped.columns:
                continue
            capped[column] = capped[column].clip(lower=lower, upper=upper)
        return capped

    def report(self, x_frame: pd.DataFrame, *, fitted_on: str) -> AppliedTreatment:
        """Count-only summary of what the fitted treatment would do (proposal/audit)."""
        self._require_fitted()
        detector = self.config.outlier_detector
        handling = self.config.outlier_handling
        if detector is OutlierDetector.NONE:
            return AppliedTreatment(
                fitted_on=fitted_on,
                detector=detector,
                handling=handling,
                rows_considered=len(x_frame),
            )
        mask = self._outlier_mask(x_frame)
        flagged = int(mask.sum())
        columns = self._affected_columns(x_frame)
        summary = AppliedTreatment(
            fitted_on=fitted_on,
            detector=detector,
            handling=handling,
            rows_considered=len(x_frame),
            affected_columns=columns,
        )
        if handling is OutlierHandling.FLAG:
            summary = summary.model_copy(update={"rows_flagged": flagged})
        elif handling is OutlierHandling.CAP:
            summary = summary.model_copy(update={"rows_capped": flagged})
        else:
            summary = summary.model_copy(update={"rows_dropped": flagged})
        return summary


def fit_and_apply(
    config: TreatmentConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series | None,
    *,
    random_seed: int = 42,
    fitted_on: str = "train",
) -> tuple[pd.DataFrame, pd.Series | None, OutlierTreatment, AppliedTreatment]:
    """Fit on training rows, then apply. Returns treated (X, y), the treatment, and the summary."""
    treatment = OutlierTreatment(config, random_seed=random_seed).fit(x_train, y_train)
    summary = treatment.report(x_train, fitted_on=fitted_on)
    x_treated, y_treated = treatment.apply(x_train, y_train)
    return x_treated, y_treated, treatment, summary
