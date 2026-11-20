# Requirements Document

Training-time data treatment — user-chosen, approved, fold-safe outlier handling and imputation,
inserted into the preprocessing chain without mutating the original data or leaking across folds.

## Introduction

Preprocessing today (`ml_engine/preprocessing/pipeline.py`) fits a scikit-learn `ColumnTransformer`
on the training fold only: for numeric columns `impute → scale`, for categoricals `impute → encode`,
with learned state (imputer statistics, scaler mean/scale, encoder categories) fitted on train and
applied unchanged to validation/test. The original dataset frame is never mutated; each step feeds
the next.

This feature adds **outlier detection and handling** as a first-class, user-chosen step at the
front of that chain, and surfaces the **imputation** and **class-weighting** choices that already
exist so the user can set them explicitly. Outlier handling is the new, risky piece: unlike
imputation and scaling it can *remove rows* or *cap values*, so it must be inserted carefully to
preserve two platform guarantees — the original data is not mutated and each stage consumes the
previous stage's output, and nothing learned touches validation or test rows.

Every treatment is **chosen and approved by the user** (handling decision (a) from the owner):
the platform detects and *proposes*, the user approves, and only then is the treatment applied.
Nothing is auto-applied. The LLM MAY propose and explain a treatment (LLM boundary B12), but it
computes no number and applies nothing without explicit user approval.

All platform invariants hold: SageMaker-only execution; S3-only artifacts; every number from the
deterministic engine; learned transforms fit on the training fold only (per fold under CV);
artifacts are versioned Pydantic contracts whose paths live only in `ml_engine/io/layout.py`;
nothing is auto-excluded; the default suite runs offline; inherited engine files change only
additively with a manifest update.

## Glossary

- **Platform**: ML Factory 2.0 as a whole.
- **Preprocessing_Chain**: the ordered sequence applied to a training fold, in this fixed order:
  **outlier handling → imputation → scaling → encoding → model fit**. Validation and test paths
  run **imputation → scaling → encoding** with the training fold's fitted state and **never**
  outlier row removal.
- **Development_Set / Test_Fold / CV_Fold / Fold_Training_Rows**: as defined in the
  cross-validation spec.
- **Outlier_Detector**: a deterministic method that identifies outlier rows from feature values.
  Candidates (design-fixed): `iqr` (per-numeric-column inter-quartile fences), `zscore`
  (standardized distance), `isolation_forest` (multivariate), and `none`.
- **Outlier_Bounds**: the fitted state of an Outlier_Detector (e.g. the per-column fences, the
  fitted isolation forest) — learned from Fold_Training_Rows only.
- **Outlier_Handling**: what is done with a detected outlier row: `flag` (record only, change
  nothing), `cap` (winsorize the offending values to the bound), or `drop` (remove the row).
- **Imputation_Strategy**: how missing values are filled (the existing `PreprocessingConfig`
  choices: numeric median/mean/constant; categorical constant/most_frequent).
- **Class_Weighting**: the existing `auto` / `none` setting plus per-model weight overrides.
- **Treatment_Config**: the user's chosen, approved data-treatment settings (Outlier_Detector,
  Outlier_Handling, Imputation_Strategy, Class_Weighting), frozen into the experiment
  configuration before training.
- **Outlier_Proposal**: a deterministic summary shown to the user before approval: per detector,
  how many rows would be flagged/capped/dropped, in which columns, with the bounds, computed on
  the Development_Set (or per fold) — so the user approves with evidence.
- **Preparation_Job / Training_Job**: the existing SageMaker stages.

## Requirements

### Requirement 1: A fixed, non-mutating preprocessing order

**User Story:** As a data scientist, I want outlier handling to run before scaling and fitting,
on the data flowing through the chain, without altering my original dataset.

#### Acceptance Criteria

1. THE Preprocessing_Chain SHALL apply steps in the fixed order outlier handling → imputation →
   scaling → encoding → model fit, so scaling statistics are computed after outliers are handled.
2. EACH step SHALL consume the output of the previous step; THE Platform SHALL NOT mutate the
   experiment's stored dataset artifact or the loaded frame in place.
3. FOR ALL inputs, running the Preprocessing_Chain SHALL leave the source dataset bytes in S3
   unchanged (non-mutation property).
4. WHEN Outlier_Handling is `none`, THE Preprocessing_Chain SHALL be byte-for-byte equivalent to
   today's chain (imputation → scaling → encoding), so existing experiments are unaffected.

### Requirement 2: Outlier handling fits on training rows only

**User Story:** As a data scientist, I want outlier bounds learned only from training rows, so no
validation or test information leaks into the model.

#### Acceptance Criteria

1. THE Platform SHALL fit Outlier_Bounds on Fold_Training_Rows only; under CV it SHALL fit them
   per CV_Fold on that fold's training rows.
2. THE Platform SHALL NOT fit Outlier_Bounds on, nor compute them from, validation rows, CV_Fold
   validation rows, or Test_Fold rows.
3. FOR ALL folds, the Outlier_Bounds fitted for a fold SHALL equal those obtained by fitting the
   detector on that fold's training rows alone (fit-isolation property).
4. FOR ALL folds, changing values only in that fold's validation rows (or Test_Fold rows) SHALL
   leave the fold's Outlier_Bounds unchanged (validation-independence property).

### Requirement 3: Row removal keeps X and y aligned and never drops held-out rows

**User Story:** As a data scientist, I want dropping outliers to remove the right rows from both
features and target, and never from data I measure on.

#### Acceptance Criteria

1. WHEN Outlier_Handling is `drop`, THE Platform SHALL remove the detected training rows from both
   the feature matrix and the target before imputation/scaling/fit, keeping them index-aligned.
2. THE Platform SHALL apply `drop` only to Fold_Training_Rows; it SHALL NEVER remove a validation
   row, a CV_Fold validation row, or a Test_Fold row.
3. WHEN scoring validation or test data, THE Platform SHALL score every row (no outlier row is
   removed from a held-out set), applying only imputation/scaling/encoding with the training
   fold's fitted state.
4. IF `drop` would remove so many training rows that a fold cannot train (below the minimum rows),
   THEN THE Preparation_Job SHALL fail with a structured error naming the counts, before any
   model fit.
5. WHEN Outlier_Handling is `cap`, THE Platform SHALL winsorize the offending values to the fitted
   bound and SHALL NOT change the row count.
6. WHEN Outlier_Handling is `flag`, THE Platform SHALL change no values and remove no rows; it
   SHALL only record which rows were flagged.

### Requirement 4: The user chooses and approves the treatment

**User Story:** As a user, I want to see what an outlier method would do and approve it, rather
than have the platform silently alter my data.

#### Acceptance Criteria

1. THE Platform SHALL expose an Outlier_Proposal computed deterministically from the
   Development_Set: per candidate detector, the count and location of rows it would flag/cap/drop
   and the bounds, so the user can judge before approving.
2. THE Platform SHALL apply a treatment only after the user approves a Treatment_Config; no
   Outlier_Handling other than `none` SHALL run without approval.
3. THE LLM MAY propose and describe an Outlier_Detector, Outlier_Handling or Imputation_Strategy
   (grounded in the deterministic counts) but SHALL NOT apply one; the choice takes effect only on
   explicit user approval (LLM boundary B12).
4. THE user SHALL be able to change one axis while holding the others fixed (e.g. keep the model
   and imputation, change only the Outlier_Detector), consistent with the experiment-variant model
   of the grid-search spec.
5. THE Treatment_Config SHALL be frozen into the experiment configuration before any job is
   submitted, so the run is reproducible from the config alone.

### Requirement 5: Imputation and class weighting are explicit choices

**User Story:** As a user, I want to set imputation and class weighting explicitly, using the
options that already exist.

#### Acceptance Criteria

1. THE Platform SHALL expose the existing Imputation_Strategy choices (numeric median/mean/
   constant; categorical constant/most_frequent) as part of the Treatment_Config, reusing
   `PreprocessingConfig`.
2. THE Platform SHALL expose the existing Class_Weighting (`auto` / `none`) and per-model weight
   overrides, reusing the current training request fields.
3. THE Platform SHALL fit any learned imputation state on the training fold only, as it does today.
4. THESE choices SHALL change no contract behavior beyond being surfaced and frozen; a run with
   the defaults SHALL match today's behavior.

### Requirement 6: Contracts, paths and backward compatibility

**User Story:** As a maintainer, I want outlier handling added through versioned, additive
contracts so existing experiments and the inherited engine stay intact.

#### Acceptance Criteria

1. THE Platform SHALL define the Outlier_Detector/Handling settings, the fitted Outlier_Bounds,
   and the Outlier_Proposal/applied-summary as versioned Pydantic contracts.
2. THE Platform SHALL add outlier fields to the frozen experiment configuration only as optional
   fields whose defaults (`none`) reproduce today's behavior.
3. THE Platform SHALL define any new artifact path only in `ml_engine/io/layout.py`.
4. WHEN an experiment configuration written before this feature is read, THE Platform SHALL treat
   it as Outlier_Handling `none` (backward-compatibility property).
5. WHERE an inherited engine file changes, THE change SHALL be additive and recorded in
   `tests/inherited_manifest.json` with a written justification. (The outlier step SHOULD be a new
   module composed around `FeaturePipeline` rather than an edit to it, if feasible.)
6. THE applied-summary (how many rows were actually flagged/capped/dropped per fold) SHALL be
   recorded in the preparation report for audit.

### Requirement 7: Determinism and reproducibility

**User Story:** As an auditor, I want outlier handling to be reproducible.

#### Acceptance Criteria

1. FOR ALL datasets and seeds, fitting an Outlier_Detector twice on the same rows with the same
   config and seed SHALL produce identical Outlier_Bounds (determinism property); `isolation_forest`
   SHALL use the experiment's random seed.
2. THE recorded Outlier_Bounds and Treatment_Config SHALL be sufficient to reproduce the applied
   treatment on the same data.
3. Serialising then parsing any new contract SHALL produce an equal object (round-trip property).

### Requirement 8: Offline, deterministic testing

**User Story:** As a developer, I want outlier handling covered offline.

#### Acceptance Criteria

1. THE test suite SHALL run with `InMemoryObjectStore`, an in-process orchestrator and mocked AWS,
   with no network.
2. THE suite SHALL include property-based tests for the non-mutation, fit-isolation,
   validation-independence, determinism and round-trip properties of Requirements 1, 2, 7.
3. THE suite SHALL include a test that `drop` removes exactly the fitted training outliers from X
   and y aligned, never a held-out row, and that validation/test rows are all scored.
4. THE suite SHALL include a test that Outlier_Handling `none` reproduces today's preprocessing
   output byte-for-byte (golden regression).
5. THE suite SHALL include a test that the fixed order holds: scaling statistics differ when
   outliers are dropped/capped first versus not, confirming outlier handling precedes scaling.

## Out of Scope

- Automatic selection of an outlier method or handling (always user-chosen and approved).
- Target/label outlier handling (this feature treats feature outliers only; regression-target
  outlier handling MAY be a later addition).
- LLM *applying* a treatment on its own (the LLM may propose and explain; the user approves — B12).
- Grid-searching outlier settings (the grid-search spec searches model parameters only; outlier
  handling is chosen manually, one setting at a time).
- Imputation methods beyond those already in `PreprocessingConfig`.

## Open Questions for the Owner

| # | Question | Working assumption |
|---|---|---|
| 1 | Which Outlier_Detectors to ship first? | `iqr`, `zscore`, `isolation_forest`, `none`; `none` default. |
| 2 | Default Outlier_Handling when a detector is chosen? | `flag` (safest: changes nothing until the user escalates to cap/drop). |
| 3 | Does the Outlier_Proposal compute on the whole Development_Set, per fold, or both? | Development_Set for the proposal the user approves; per-fold bounds are what actually fit at train time (Req 2). Report both counts. |
| 4 | Should `cap` winsorize to the training bound or to a percentile? | To the fitted training bound of the chosen detector, for consistency with detection. |
| 5 | Multivariate (`isolation_forest`) vs per-column (`iqr`/`zscore`) interaction with `cap`. | `cap` applies only to per-column detectors; with `isolation_forest`, handling is `flag` or `drop` (you cannot cap a multivariate outlier to a single column bound). |
