# Design: Training-time data treatment

## Overview

Add **outlier handling** as a user-chosen, approved, fold-safe step at the front of the
preprocessing chain, and surface the imputation and class-weighting choices that already exist.
The order becomes **outlier handling → imputation → scaling → encoding → model fit** on each
training fold; validation and test rows run **imputation → scaling → encoding** with the training
fold's fitted state and are never outlier-removed.

The central design fact, from reading the code: the existing `FeaturePipeline` is a scikit-learn
`ColumnTransformer` that fits on training rows and operates **column-wise** — it cannot drop rows
or touch the target. Outlier handling is **row-level** and must happen on `(X_train, y_train)`
*before* the pipeline fits, so the imputer/scaler see post-outlier data. Therefore the outlier step
is a **separate fitted object** (`OutlierTreatment`), composed around `FeaturePipeline` rather than
edited into it (as the requirements suggest), fitted per fold on training rows only, applied to the
training frame only, and recorded for audit and reproducibility.

When the treatment is `none` (the default), the chain is byte-for-byte today's chain, so every
existing experiment is unaffected.

## Architecture

```
preparation (per fold / per training side):
  training rows ──▶ OutlierTreatment.fit(X_train, y_train)      # bounds from train rows only
                     │
                     ├─ drop: remove detected training rows (X and y aligned)
                     ├─ cap : winsorise offending values to the fitted bound
                     └─ flag: record only
                     ▼
                 treated training rows ──▶ FeaturePipeline.fit  # impute → scale → encode
                     │                         (on treated rows, so scaling follows outliers)
                     ▼
                 write the treated train frame + the fitted pipeline + the OutlierTreatment

validation / test:
  held-out rows ──▶ FeaturePipeline.transform  (impute → scale → encode); NO outlier removal
```

Outlier handling lives entirely in **preparation** (the one place a learned transform is fitted),
so it is naturally per-fold under CV and never sees held-out rows. The training stage is unchanged:
it reads the already-treated training frame and the fitted pipeline and transforms.

## Components and Interfaces

### C1. Treatment contracts (`ml_engine/contracts/treatment.py`, NEW)

- `OutlierDetector` enum: `none`, `iqr`, `zscore`, `isolation_forest`.
- `OutlierHandling` enum: `flag`, `cap`, `drop`.
- `TreatmentConfig` (versioned): `outlier_detector` (default `none`), `outlier_handling` (default
  `flag`), `zscore_threshold`/`iqr_multiplier`/`isolation_contamination` (detector knobs with safe
  defaults), and a reference to the existing `PreprocessingConfig` imputation + `ClassWeighting`
  (those are not duplicated; they are surfaced, Req 5). Frozen into `ExperimentConfig.treatment`
  (optional, `exclude_if` at the `none` default so pre-feature configs stay byte-identical,
  Req 6.2, 6.4).
- `OutlierBounds` (versioned): the fitted state — per-column fences (`iqr`/`zscore`) or the fitted
  `IsolationForest` parameters/seed (`isolation_forest`) — enough to reproduce the treatment on the
  same rows (Req 7.2).
- `OutlierProposal` (versioned): per candidate detector, the count and columns of rows it would
  flag/cap/drop and the bounds, computed deterministically on the Development_Set (Req 4.1).
- `AppliedTreatment` (versioned): the per-fold applied summary (rows flagged/capped/dropped,
  columns), recorded in the preparation report for audit (Req 6.6).

### C2. The outlier engine (`ml_engine/preprocessing/outliers.py`, NEW)

- `OutlierTreatment` with `fit(X_train, y_train) -> self`, `apply(X, y) -> (X', y')` and
  `bounds() -> OutlierBounds`:
  - `fit` computes bounds from training rows only (per-column fences for `iqr`/`zscore`; a seeded
    `IsolationForest` for `isolation_forest`). Deterministic: the same rows + config + seed give
    identical bounds (Req 7.1); `isolation_forest` uses the experiment's `random_seed`.
  - `apply` on `flag`: returns `(X, y)` unchanged and records the flagged row ids. On `cap`:
    winsorises the offending numeric values to the fitted bound; row count unchanged (Req 3.5). On
    `drop`: removes the detected rows from both X and y, index-aligned (Req 3.1), and raises a
    structured `OutlierTreatmentError` if too few training rows remain (Req 3.4).
  - `cap` applies only to per-column detectors (`iqr`/`zscore`); with `isolation_forest`, handling
    is `flag` or `drop` (open question 5) — a `cap` + `isolation_forest` config is refused at
    validation time.
- A `none` treatment is a pass-through: `fit`/`apply` return the frame unchanged (Req 1.4).
- Pure `ml_engine`; no IO, no AWS.

### C3. Preparation: fit and apply per fold (`jobs/preprocessing/`)

The outlier step slots into the two existing fit paths, before `FeaturePipeline.fit`:

- **Single split** (`jobs/preprocessing/main.py` and the Test_Fold path in `cv.py::_prepare_single`):
  fit `OutlierTreatment` on the training frame, apply it (drop/cap) to get the treated training
  frame, then fit the pipeline on the treated frame and write the treated `train.parquet`. The
  validation frame is written untouched.
- **CV** (`cv.py::_prepare_cv` + `ml_engine/preprocessing/folds.py::fit_fold_pipelines`): because
  each fold's bounds differ, the treatment is fit and applied **per fold** on that fold's training
  rows, then that fold's pipeline is fit on the treated rows. The per-fold `OutlierTreatment` is
  saved next to the per-fold preprocessor (`preprocessing/folds/fold_{i}/outlier_{strategy}.joblib`
  or a single `outliers.joblib` per fold). The shared `development.parquet` stays the untreated
  Development_Set (each fold applies its own treatment at train time, C4); the treatment is not
  baked into it, since drop/cap differ per fold.
- The preparation report gains the `AppliedTreatment` summary per fold (Req 6.6).

### C4. Training: apply the fold's treatment before fitting the model (`jobs/training/`)

Under CV, the fold loop (`jobs/training/cv.py`) already loads `development.parquet` and each fold's
preprocessor. It additionally loads the fold's `OutlierTreatment`, applies it to the fold's
training rows (drop/cap) before `pipeline.transform`, and fits the model on the treated rows. The
fold's **validation** rows are transformed untreated and scored on every row (Req 3.3). Single
split needs no training-stage change (the treated `train.parquet` was already written in
preparation). The engine `train_model` is unchanged — it receives an already-treated, already-fitted
pipeline and frame.

### C5. The Outlier_Proposal (`backend/services` + a preparation/side computation)

- A deterministic `OutlierProposal` is computed on the Development_Set for each candidate detector
  (counts + bounds), surfaced so the user approves with evidence (Req 4.1). It is computed by the
  engine (`OutlierTreatment.fit` on the Development_Set, reporting counts without applying), run in
  a Processing job or folded into EDA/preparation output. No number comes from the LLM.
- Approval is the existing propose/approve shape: the user sets `TreatmentConfig` on the training
  request; the LLM may suggest it but the user submits it (Req 4.2, 4.3, B12). Changing one axis
  (detector) while holding others (model, imputation) is just a different `TreatmentConfig` on the
  next run (Req 4.4), consistent with the grid-search experiment-variant model.

### C6. Request + config surfacing

- `TrainingRequest` gains optional treatment fields (`outlier_detector`, `outlier_handling`,
  detector knobs); absence = `none` = today (Req 6.2). Imputation (`PreprocessingConfig`) and
  `class_weighting` are already on the request; they are surfaced in the UI/MCP, not re-added
  (Req 5).
- `ExperimentService` freezes `TreatmentConfig` into `ExperimentConfig.treatment` before
  submission (Req 4.5), validating the detector/handling combination (refusing `cap` +
  `isolation_forest`).

## Data Models

New versioned contracts in `ml_engine/contracts/treatment.py`: `OutlierDetector`, `OutlierHandling`,
`TreatmentConfig`, `OutlierBounds`, `OutlierProposal`, `AppliedTreatment`. `ExperimentConfig` gains
an optional `treatment: TreatmentConfig | None` (excluded at `none`). `PreparationReport` gains an
optional per-fold `applied_treatment` list. `PreprocessingConfig` and `ClassWeighting` are reused
unchanged. New layout paths: the per-fold and single-split `OutlierTreatment` artifact. No existing
path or default changes, so a `none` run is byte-identical.

## Correctness Properties

### Property 1: non-mutation
Running the chain leaves the source dataset bytes in S3 unchanged and does not mutate the loaded
frame in place.
**Validates: Requirements 1.2, 1.3**

### Property 2: fit isolation
A fold's `OutlierBounds` equal those from fitting the detector on that fold's training rows alone.
**Validates: Requirements 2.1, 2.3**

### Property 3: validation independence
Changing values only in a fold's validation rows (or Test_Fold rows) leaves the fold's
`OutlierBounds` unchanged.
**Validates: Requirements 2.2, 2.4**

### Property 4: aligned, held-out-safe drop
`drop` removes exactly the fitted training outliers from X and y aligned, never a validation or
Test_Fold row; held-out rows are all scored.
**Validates: Requirements 3.1, 3.2, 3.3**

### Property 5: determinism and round trip
The same rows + config + seed produce identical bounds; serialising then parsing any new contract
yields an equal object.
**Validates: Requirements 7.1, 7.3**

### Property 6: none is today
`outlier_handling = none` reproduces today's preprocessing output byte-for-byte.
**Validates: Requirements 1.4, 6.4**

## Error Handling

- A `drop` that leaves a fold below the minimum training rows raises `OutlierTreatmentError` ->
  `JobError("OUTLIER_LEFT_TOO_FEW_ROWS", ...)` in preparation, before any model fit (Req 3.4),
  naming the counts.
- `cap` + `isolation_forest` is refused at the service layer with a structured error (open q 5).
- An unknown detector/handling value is a `ValidationError` before submission.

## Testing Strategy

Offline, with `InMemoryObjectStore` + `InlineOrchestrator` + mocked AWS; property tests via
Hypothesis.

- Property tests for non-mutation, fit-isolation, validation-independence, determinism, round-trip
  (Req 8.2).
- `drop` removes exactly the fitted training outliers from X and y aligned, never a held-out row,
  and validation/test rows are all scored (Req 8.3).
- `none` reproduces today's preprocessing output byte-for-byte (golden regression, Req 8.4).
- The fixed order holds: scaling statistics differ when outliers are dropped/capped first vs not,
  confirming outlier handling precedes scaling (Req 8.5).
- Per-fold isolation under CV: each fold's bounds come from its own training rows.

## What changes in inherited files (manifest)

| File | Change | Justification |
|---|---|---|
| `ml_engine/contracts/config.py` | Add `ExperimentConfig.treatment` (optional, excluded at none) | Freeze the approved treatment; absence reproduces today |
| `ml_engine/contracts/preparation.py` | Add optional `applied_treatment` per-fold summary | Audit of what was flagged/capped/dropped (Req 6.6) |
| `ml_engine/preprocessing/folds.py` | Fit + apply the `OutlierTreatment` per fold before fitting the pipeline | The per-fold, fold-safe insertion point (Req 2) |
| `jobs/preprocessing/main.py`, `jobs/preprocessing/cv.py` | Fit/apply the treatment before `FeaturePipeline.fit`; write the treated train frame; record the summary | The single place a learned transform is fitted |
| `jobs/training/cv.py` | Apply the fold's treatment to its training rows before the fold model fits | drop/cap differ per fold, applied at train time |
| `ml_engine/io/layout.py` | Add the `OutlierTreatment` artifact paths | New additive paths, defined only here |
| `backend/schemas.py`, `backend/services/experiments.py` | Optional treatment request fields + freeze into config | Surface and freeze the approved treatment |

Each inherited edit is additive and recorded in `tests/inherited_manifest.json` with a written
justification. New modules (`contracts/treatment.py`, `preprocessing/outliers.py`) are registered
as NEW. The outlier step is composed around `FeaturePipeline`, not an edit to it (Req 6.5).

## Open questions (resolved in design)

- **Detectors (Q1):** `iqr`, `zscore`, `isolation_forest`, `none`; `none` default.
- **Default handling (Q2):** `flag` when a detector is chosen.
- **Proposal scope (Q3):** proposal counts on the Development_Set; per-fold bounds are what fit at
  train time; report both.
- **Cap target (Q4):** winsorise to the fitted training bound.
- **Isolation forest + cap (Q5):** `cap` is per-column only; `isolation_forest` supports `flag`/
  `drop`; `cap` + `isolation_forest` is refused.
