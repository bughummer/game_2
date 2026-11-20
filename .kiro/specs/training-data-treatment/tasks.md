# Implementation Plan: Training-time data treatment

## Overview

Python, following `design.md`. The core insight: outlier handling is **row-level** and must happen
on `(X_train, y_train)` *before* `FeaturePipeline` fits, so it is a **separate fitted object**
(`OutlierTreatment`) composed around the pipeline, not an edit to it. Build bottom-up — treatment
contracts, the pure outlier engine (+ PBT), then the per-fold fit/apply in preparation, the CV
train-time apply, then the proposal and request surfacing, then the offline tests. Every
inherited-file change is additive and followed by a manifest update. When the treatment is `none`
(the default), the chain is byte-for-byte today's chain. Commands:
`.venv\Scripts\python -m pytest -q -p no:cacheprovider`, `.venv\Scripts\python -m ruff check .`,
`.venv\Scripts\python -m ruff format --check .`.

## Task Dependency Graph

```
1 treatment contracts (NEW: contracts/treatment.py) + ExperimentConfig.treatment (inherited)
└─▶ 2 outlier engine (NEW: preprocessing/outliers.py) + PBT
     └─▶ 3 layout paths (inherited) + preparation fit/apply per fold (inherited: folds, prep jobs)
          └─▶ 4 CV train-time apply (inherited: training/cv.py)
               └─▶ 5 checkpoint
                    └─▶ 6 proposal + request surfacing + freeze/validate (inherited: schemas, service)
                         └─▶ 7 end-to-end + golden/backward-compat + order tests
                              └─▶ 8 final checkpoint
```

```json
{
  "waves": [
    { "wave": 1, "tasks": ["1"], "depends_on": [] },
    { "wave": 2, "tasks": ["2"], "depends_on": ["1"] },
    { "wave": 3, "tasks": ["3"], "depends_on": ["2"] },
    { "wave": 4, "tasks": ["4"], "depends_on": ["3"] },
    { "wave": 5, "tasks": ["5"], "depends_on": ["4"] },
    { "wave": 6, "tasks": ["6"], "depends_on": ["5"] },
    { "wave": 7, "tasks": ["7"], "depends_on": ["6"] },
    { "wave": 8, "tasks": ["8"], "depends_on": ["7"] }
  ]
}
```

## Tasks

- [x] 1. Treatment contracts and the frozen config field (design C1; inherited: config)
  - `ml_engine/contracts/treatment.py` (NEW): `OutlierDetector` enum (`none`, `iqr`, `zscore`,
    `isolation_forest`), `OutlierHandling` enum (`flag`, `cap`, `drop`); `TreatmentConfig`
    (versioned: `outlier_detector` default `none`, `outlier_handling` default `flag`,
    `zscore_threshold`/`iqr_multiplier`/`isolation_contamination` knobs with safe defaults; it
    references but does not duplicate `PreprocessingConfig` imputation + `ClassWeighting`);
    `OutlierBounds` (versioned fitted state — per-column fences or the seeded isolation-forest
    params); `OutlierProposal` (versioned per-detector count/columns/bounds); `AppliedTreatment`
    (versioned per-fold applied summary: rows flagged/capped/dropped, columns)
  - `ml_engine/contracts/config.py`: add `ExperimentConfig.treatment: TreatmentConfig | None = None`,
    excluded at the `none` default so pre-feature configs stay byte-identical (Req 6.2, 6.4)
  - Register `contracts/treatment.py` as NEW; manifest update + justification for `config.py`;
    regenerate
  - Tests: every new contract round-trips (serialise→parse equal, Req 7.3); a config with no
    treatment reads as `none` (backward compat, Req 6.4); the `none` default is excluded from the
    serialised config
  - _Requirements: 6.1, 6.2, 6.4, 7.3_

- [x] 2. The outlier engine (design C2; NEW: preprocessing/outliers.py)
  - `ml_engine/preprocessing/outliers.py` (NEW): `OutlierTreatment` with
    `fit(X_train, y_train) -> self`, `apply(X, y) -> (X', y')`, `bounds() -> OutlierBounds`,
    and a count-only `report(X) -> AppliedTreatment`-shaped summary for the proposal
  - `fit` computes bounds from training rows only — per-column IQR/zscore fences, or a seeded
    `IsolationForest` using the experiment `random_seed` (deterministic: same rows+config+seed →
    identical bounds, Req 7.1)
  - `apply`: `flag` returns `(X, y)` unchanged and records flagged ids; `cap` winsorises offending
    numeric values to the fitted bound, row count unchanged (Req 3.5), per-column detectors only;
    `drop` removes detected rows from X and y index-aligned (Req 3.1) and raises
    `OutlierTreatmentError` if too few training rows remain (Req 3.4)
  - `none` is a pass-through (Req 1.4); `cap` + `isolation_forest` is refused with a structured
    error (open q 5). Pure `ml_engine`, no IO, no AWS
  - Hypothesis property tests (`max_examples=100`, `deadline=None`, `derandomize=True`, tagged
    `# Feature: training-data-treatment, Property N`): non-mutation (P1), fit-isolation (P2),
    validation-independence (P3), determinism + round-trip (P5), aligned held-out-safe drop (P4)
  - _Requirements: 1.4, 2.1, 2.3, 2.4, 3.1, 3.4, 3.5, 3.6, 7.1, 7.2, 8.2_

- [x] 3. Layout paths + preparation fit/apply per fold (design C3; inherited: layout, folds, prep jobs)
  - `ml_engine/io/layout.py`: add the single-split and per-fold `OutlierTreatment` artifact paths
    (e.g. `preprocessing/outliers.joblib`, `preprocessing/folds/fold_{i}/outliers.joblib`) —
    additive, defined only here (Req 6.3)
  - `ml_engine/contracts/preparation.py`: add optional per-fold `applied_treatment:
    list[AppliedTreatment]` to `PreparationReport` (Req 6.6)
  - `ml_engine/preprocessing/folds.py`: in `fit_fold_pipelines`, fit `OutlierTreatment` on each
    fold's training rows and apply (drop/cap) before `FeaturePipeline.fit`, so the imputer/scaler
    see post-outlier data; return/save the per-fold treatment (Req 2.1)
  - `jobs/preprocessing/main.py` + `jobs/preprocessing/cv.py`: fit/apply the treatment before the
    pipeline fit; write the treated `train.parquet` (single split); save the per-fold treatment
    artifact; leave the shared `development.parquet` untreated (each fold applies its own at train
    time); record the `AppliedTreatment` summary in the report. Validation/test frames written
    untouched
  - Manifest updates + justifications for `layout.py`, `preparation.py`, `folds.py`,
    `jobs/preprocessing/main.py`, `jobs/preprocessing/cv.py`; regenerate
  - Tests: single-split writes a treated train frame and a treatment artifact; the report carries
    the applied summary; held-out frames are byte-unchanged; a `drop` leaving too few rows fails
    with `OUTLIER_LEFT_TOO_FEW_ROWS` before any fit (Req 3.4)
  - _Requirements: 2.1, 2.2, 3.2, 3.4, 6.3, 6.6_

- [x] 4. CV train-time apply (design C4; inherited: training/cv.py)
  - `jobs/training/cv.py`: in the fold loop, load each fold's `OutlierTreatment` alongside the
    fold preprocessor, apply it (drop/cap) to the fold's training rows before `pipeline.transform`,
    and fit the model on the treated rows; the fold's **validation** rows are transformed untreated
    and scored on every row (Req 3.3). Single split needs no training-stage change (the treated
    `train.parquet` was written in preparation). The engine `train_model` is unchanged
  - Manifest update + justification for `jobs/training/cv.py`; regenerate
  - Tests: under CV, each fold trains on its own treated rows; every validation row is scored
    (none removed); with `none`, the CV path is byte-identical to today
  - _Requirements: 2.1, 3.3_

- [x] 5. Checkpoint - Ensure all tests pass
  - `.venv\Scripts\python -m pytest -q -p no:cacheprovider`, `ruff check .`, `ruff format --check .`;
    confirm `tests/test_inherited_manifest.py` passes with the regenerated manifest; ask the user
    if questions arise
  - Result: contracts + engine + preparation + CV train-time apply in place; ruff clean; manifest
    regenerated (tasks 1-4)

- [x] 6. Proposal, request surfacing, freeze + validate (design C5, C6; inherited: schemas, service)
  - Compute a deterministic `OutlierProposal` on the Development_Set per candidate detector
    (counts + bounds, no apply), via `OutlierTreatment.report`, surfaced so the user approves with
    evidence (Req 4.1); no number comes from the LLM
  - `backend/schemas.py`: `TrainingRequest` gains optional treatment fields (`outlier_detector`,
    `outlier_handling`, detector knobs); absence = `none` = today (Req 6.2). Imputation
    (`PreprocessingConfig`) and `class_weighting` are already on the request — surfaced, not re-added
    (Req 5)
  - `backend/services/experiments.py`: freeze `TreatmentConfig` into `ExperimentConfig.treatment`
    before submission (Req 4.5); validate the detector/handling combination (refuse `cap` +
    `isolation_forest`; reject unknown detector/handling with a `ValidationError` before submission)
  - `start_training` (MCP tool + API route) echoes the resolved treatment (chosen detector/handling)
    alongside the evaluation decision; the LLM may suggest it, the user submits it (Req 4.2, 4.3, B12)
  - Manifest updates + justifications for `schemas.py`, `experiments.py` if hash-tracked; regenerate
  - Tests: a request with treatment freezes it into the config; `cap`+`isolation_forest` refused
    with a structured error before submission; changing only the detector (holding model/imputation)
    is a different `TreatmentConfig` (Req 4.4); the echoed start_training response reports the
    treatment
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 6.2_

- [x] 7. End-to-end, golden/backward-compat and fixed-order tests (design Testing Strategy)
  - End-to-end offline (InMemoryObjectStore + InlineOrchestrator + mocked AWS): a single-split and
    a CV run with `drop` and with `cap`; assert the treated training rows, the per-fold bounds from
    each fold's own rows, every held-out row scored, and the applied summary recorded
  - Golden regression: `outlier_handling = none` reproduces today's preprocessing output
    byte-for-byte (Req 8.4)
  - Fixed-order test: scaling statistics differ when outliers are dropped/capped first versus not,
    confirming outlier handling precedes scaling (Req 8.5)
  - Drop correctness: `drop` removes exactly the fitted training outliers from X and y aligned,
    never a held-out row, and validation/test rows are all scored (Req 8.3)
  - _Requirements: 1.1, 3.3, 8.3, 8.4, 8.5_

- [x] 8. Final checkpoint - Ensure all tests pass
  - Run `.venv\Scripts\python -m pytest -q -p no:cacheprovider`, `.venv\Scripts\python -m ruff check .`,
    `.venv\Scripts\python -m ruff format --check .`; confirm `tests/test_inherited_manifest.py`
    and `tests/test_architecture_boundaries.py` pass; ensure all pass; ask the user if questions arise
  - Result: 1485 passed, 1 skipped; ruff check + format clean; manifest + architecture-boundary
    tests pass

## Notes

- Outlier handling is a separate fitted object composed around `FeaturePipeline`, not an edit to it
  (the ColumnTransformer is column-wise and cannot drop rows or touch the target). This keeps the
  inherited pipeline untouched and satisfies Req 6.5.
- The shared `development.parquet` stays untreated under CV; each fold's drop/cap differs, so the
  per-fold treatment is saved in preparation and applied at train time in `jobs/training/cv.py`.
- A `none` treatment is byte-for-byte today's chain, so every existing experiment is unaffected.
- Property tests use Hypothesis (`max_examples=100`, `deadline=None`, `derandomize=True`), tagged
  `# Feature: training-data-treatment, Property N`.
