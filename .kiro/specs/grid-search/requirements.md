# Requirements Document

Grid search over model parameters — a user-specified, user-approved, guard-bounded search over
model hyperparameters, where every grid point is trained, saved as its own model version, and
compared, and nothing is auto-selected.

## Introduction

Training today fits each enabled model once with one set of hyperparameters (the plugin defaults
plus any per-model overrides) and compares the models. This feature adds classical **grid search**
over **model parameters only**: the user gives value lists per parameter (e.g. XGBoost
`max_depth ∈ {4, 6, 8}`, `reg_alpha ∈ {0, 0.1}`), the platform expands the cross-product into grid
points, trains each point, and compares them. Each grid point is saved as an independent model
version so any of them can be validated later (the model-version store defined in the control-UI
spec).

Scope is deliberately narrow (owner decisions): the grid searches **model hyperparameters only**.
The feature set, the data split, outlier handling and imputation are held **fixed** across a grid
run and chosen manually (outlier/imputation via the training-data-treatment spec). The user
controls the parameter set to search and the number of folds. The expanded grid is shown with its
fit count and **approved by the user** before anything runs, and the existing guards (fold-fit
ceiling, budget, concurrency) bound it — a grid that would exceed a ceiling is refused with the
number, never silently trimmed. The platform ranks the grid points; it never auto-applies a
"winner".

This moves a previously out-of-scope boundary: the ml-factory-2 spec listed "no HPO". That line is
amended to allow **user-specified, user-approved grid search**; automated/Bayesian search remains
out of scope.

All platform invariants hold: SageMaker-only execution; S3-only artifacts; every number from the
deterministic engine; CV folds and preprocessing fit on training rows only (per fold); nothing is
auto-excluded; artifacts are versioned Pydantic contracts whose paths live only in
`ml_engine/io/layout.py`; ownership and audit on every mutation; the default suite runs offline;
inherited engine files change only additively with a manifest update.

## Glossary

- **Platform**: ML Factory 2.0 as a whole.
- **Model**: one algorithm plugin (logistic_regression, elastic_net, random_forest, xgboost, and
  their regressor variants), each exposing a known parameter space.
- **Parameter_Grid**: a mapping, per Model, of parameter name → list of candidate values, supplied
  by the user (e.g. `{"xgboost": {"max_depth": [4, 6, 8], "reg_alpha": [0, 0.1]}}`).
- **Grid_Point**: one fully-specified parameter combination for one Model — one cell of the
  cross-product of that Model's Parameter_Grid.
- **Grid_Plan**: the full expanded list of Grid_Points for a request, with the per-point and total
  **fit count** (Grid_Points × (K + 1) under CV, or × 1 in Single_Split_Mode), shown to the user
  for approval before anything runs.
- **Fit_Count_Ceiling**: the existing `cv_max_fits` setting (and the model-per-run/concurrency
  guards) that bound how many fits one request may schedule.
- **Model_Version**: a trained, saved model artifact plus metadata with a stable id (as defined in
  the control-UI spec). Every Grid_Point produces exactly one Model_Version.
- **K / CV_Mode / Single_Split_Mode**: as defined in the cross-validation spec.
- **Best_By_CV_Mean**: the ranking the evaluation stage already uses (CV mean of the primary
  metric, then lower std, then id). Grid_Points are ranked the same way.

## Requirements

### Requirement 1: Specify a grid over model parameters only

**User Story:** As a data scientist, I want to give value lists for model hyperparameters and have
every combination tried, so that I can tune classically.

#### Acceptance Criteria

1. THE Platform SHALL accept a Parameter_Grid per Model as part of a training request: parameter
   name → list of candidate values, for parameters the Model's plugin exposes.
2. THE Platform SHALL expand each Model's Parameter_Grid into the cross-product of its parameters
   to produce the Grid_Points for that Model.
3. THE grid SHALL cover model hyperparameters only; THE Platform SHALL reject a grid that tries to
   vary the feature set, the split, outlier handling or imputation, naming the offending key.
4. THE Platform SHALL validate every grid value against the plugin's parameter space before
   submission, rejecting an unknown parameter or an invalid value with a structured error.
5. WHEN a Model has no Parameter_Grid entry, THE Platform SHALL train it once with its defaults, as
   today (a grid request does not force every model into a grid).
6. THE user SHALL control the number of folds K for the run, within the configured bounds.

### Requirement 2: Hold everything except model parameters fixed

**User Story:** As a data scientist, I want a fair comparison, so that score differences come only
from the parameters.

#### Acceptance Criteria

1. ACROSS all Grid_Points of a request, THE Platform SHALL use the same feature set, the same
   fold assignment / split, the same outlier handling and the same imputation.
2. THE Platform SHALL fit preprocessing per fold on training rows only, identically for every
   Grid_Point, so preprocessing is not a variable of the search.
3. THE Platform SHALL change no data-treatment setting as part of expanding or running a grid.

### Requirement 3: Show the grid and require approval before running

**User Story:** As a user, I want to see how many fits a grid will cost and approve it, so that a
large grid cannot run or cost without my say-so.

#### Acceptance Criteria

1. THE Platform SHALL compute a Grid_Plan — the full list of Grid_Points and the total fit count
   (Grid_Points × (K + 1) in CV_Mode, × 1 otherwise) — and SHALL expose it before any job runs.
2. THE Platform SHALL require explicit user approval of the Grid_Plan before submitting any
   training job.
3. THE LLM MAY propose and describe a Grid_Plan (including suggesting parameter values) but SHALL
   NOT approve it and SHALL NOT submit it; the values and the approval are the user's (LLM boundary
   B12: propose freely, apply nothing without approval).
4. THE audit trail SHALL record the approved Grid_Plan (the parameters searched, K, the fit count).

### Requirement 4: Enforce the fit-count, budget and concurrency guards

**User Story:** As a budget owner, I want a grid bounded by the same guards as any training run,
so that it cannot silently multiply cost.

#### Acceptance Criteria

1. IF the Grid_Plan's total fit count exceeds the Fit_Count_Ceiling (`cv_max_fits`), THEN THE
   Platform SHALL reject the request before submission and return the requested and allowed fit
   counts.
2. THE Platform SHALL NOT silently trim a grid to fit a ceiling; it SHALL refuse and let the user
   shrink the grid, lower K, or raise the ceiling deliberately.
3. THE Platform SHALL run Grid_Point training jobs within the existing per-experiment and
   platform-wide concurrency caps, in waves, exactly as multi-model training already does.
4. WHERE budget enforcement is on, THE Platform SHALL estimate the grid's cost across all fits and
   enforce the per-experiment and monthly ceilings before submission, refusing with the estimate
   and the ceiling if exceeded.
5. THE guards SHALL run in the service layer before any job is submitted, so no partial grid is
   left running when the request is refused.

### Requirement 5: Train and save every grid point as a model version

**User Story:** As a user, I want every grid point kept as its own model, so that I can validate
any of them later, not just the best.

#### Acceptance Criteria

1. THE Platform SHALL train each Grid_Point (under CV_Mode when the experiment uses it, else a
   single split), reusing the existing training and fold machinery unchanged.
2. THE Platform SHALL save each Grid_Point as one Model_Version in the model-version store, with
   its exact parameters, metrics and training context, at a path defined only in
   `ml_engine/io/layout.py`; no Grid_Point SHALL overwrite another's artifacts.
3. A single Grid_Point failure SHALL be recorded for that point and SHALL NOT abort the other
   points (per-point isolation, as per-model isolation works today).
4. EACH Grid_Point Model_Version SHALL be independently loadable and independently validatable on
   an uploaded dataset (per the control-UI validation flow).
5. THE Model_Version of a Grid_Point SHALL record which grid and which combination produced it, so
   results are traceable to the Parameter_Grid.

### Requirement 6: Compare and rank grid points, never auto-select

**User Story:** As a decision maker, I want the grid points ranked with the spread shown, and I
decide which to use.

#### Acceptance Criteria

1. THE Platform SHALL rank Grid_Points by Best_By_CV_Mean (CV mean of the primary metric, then
   lower std, then id), reusing the existing CV ranking.
2. THE Platform SHALL present the full grid results — every point's parameters and metrics (mean ±
   std per metric under CV) — not only the top point.
3. THE Platform SHALL NOT auto-apply or auto-promote any Grid_Point; selecting one for downstream
   use SHALL be an explicit user action.
4. THE instability and outlier-fold warnings SHALL apply to Grid_Points as to any CV run.
5. WHEN two Grid_Points are within the statistical-tie margin on the primary metric, THE Platform
   SHALL surface that, so the user can prefer the simpler parameters.

### Requirement 7: Contracts, paths and backward compatibility

**User Story:** As a maintainer, I want grid search added through versioned, additive contracts.

#### Acceptance Criteria

1. THE Platform SHALL define the Parameter_Grid, Grid_Plan and grid-result contracts as versioned
   Pydantic contracts.
2. THE Platform SHALL add grid fields to the training request / frozen configuration only as
   optional fields whose absence reproduces today's single-combination training.
3. THE Platform SHALL define every new artifact path only in `ml_engine/io/layout.py`.
4. WHERE an inherited engine file changes, THE change SHALL be additive and recorded in
   `tests/inherited_manifest.json` with a written justification.
5. THE ml-factory-2 "out of scope: HPO" boundary SHALL be amended to permit user-specified,
   user-approved grid search while keeping automated/Bayesian search out of scope.

### Requirement 8: Offline, deterministic testing

**User Story:** As a developer, I want grid search covered offline.

#### Acceptance Criteria

1. THE test suite SHALL run with `InMemoryObjectStore`, an in-process orchestrator and mocked AWS,
   with no network.
2. THE suite SHALL include a property-based test that grid expansion equals the cross-product of
   the per-parameter value lists, and that the reported fit count equals Grid_Points × (K + 1)
   (CV) or Grid_Points × 1 (single split).
3. THE suite SHALL include a test that a grid over the Fit_Count_Ceiling is refused with the
   counts and submits no job, and that a grid varying a non-parameter key is rejected.
4. THE suite SHALL include a test that every Grid_Point becomes a distinct, loadable Model_Version
   and that one point's failure leaves the others intact.
5. THE suite SHALL include a test that a training request with no Parameter_Grid behaves exactly
   as today (single combination per model).

## Out of Scope

- Automated or Bayesian hyperparameter optimisation, early-stopping of the search, successive
  halving — only exhaustive user-specified grids.
- Searching the feature set, split, outlier handling or imputation (held fixed; chosen manually).
- Nested cross-validation.
- Auto-selecting or auto-deploying the best grid point.

## Open Questions for the Owner

| # | Question | Working assumption |
|---|---|---|
| 1 | Does a grid run under whatever evaluation mode the experiment is in (CV or single split), or force CV? | Run under the experiment's mode; grids are most useful under CV but single-split grids are allowed (fit count × 1). |
| 2 | How are Grid_Points named as model versions? | `<model>#<short-hash-of-params>` plus the full parameter dict in the version record, so names are stable and readable. |
| 3 | Per-parameter value-count cap, to keep grids sane before the fit-count check? | No separate cap; the Fit_Count_Ceiling is the single guard. |
| 4 | Should the grid results feed the model-comparison report, or a separate grid report? | Extend the comparison to list grid points; a dedicated grid view in the UI. |
| 5 | Do grid-point model versions count against retention/storage limits differently? | No; they are ordinary Model_Versions under the experiment prefix, subject to the same retention. |
