# Design: Grid search over model parameters

## Overview

Grid search expands a user-supplied, user-approved grid of model hyperparameters into one
**grid point per parameter combination**, trains each point, saves it as its own model version,
and ranks them — reusing the existing training, CV, evaluation and model-version machinery almost
untouched. The central design choice, confirmed by reading the code: **a grid point is an ordinary
`ModelSpec`** with a distinct name and populated `params`. Expanding a grid into N `ModelSpec`s
turns into N training jobs, N `models/<name>/metadata.json` artifacts, N ranked rows, and N model
versions, with no change to the orchestrator, the training stage, the fold loop, the evaluation
ranking, or the version store.

Three things genuinely need adding: (1) a **grid id scheme** and a base-plugin lookup so a grid
point's name resolves to its base plugin; (2) a **parameter-space declaration** on the plugin
contract so grid values can be validated before submission (Req 1.4); (3) the **grid expansion +
Grid_Plan + approval** at the service layer. Everything else is reuse.

## Architecture

```
TrainingRequest.parameter_grid  ─┐
                                 ├─▶ ExperimentService: expand grid → list[ModelSpec]  (grid points)
                                 │     • validate each value vs the plugin's parameter_space (Req 1.4)
                                 │     • reject non-parameter keys (Req 1.3)
                                 │     • Grid_Plan (points + fit count) → require approval (Req 3)
                                 │     • fit-count / budget / concurrency guards (Req 4)
                                 ▼
          ExperimentConfig.models = [grid points]  ──▶ orchestrator: one training job per point
                                 │                       (unchanged; names flow to models/<name>/)
                                 ▼
          training (single or CV) per point ──▶ evaluation ranks all points ──▶ model versions
```

The grid lives on the request and is expanded into `config.models` before freezing, so every
downstream stage sees a normal multi-model run. All platform invariants hold unchanged.

## Components and Interfaces

### C1. Grid id scheme and base-plugin lookup (`ml_engine/models/registry.py`)

- A grid point's name is `<base>__<hash8>` where `<base>` is the registered plugin name and
  `<hash8>` is the first 8 hex chars of a sha256 over the point's sorted params. The double
  underscore is URL/S3-safe and survives SageMaker job-name sanitisation (`[a-zA-Z0-9-]`) without
  colliding — unlike `#`, which is reserved in URIs. The human-readable `<base>#…` form (open
  question 2) is display-only, carried in the version record, never a path.
- Add `base_model_name(name) -> str` to the registry: returns the substring before `__`, or the
  name unchanged. `get_plugin` resolves a grid-point name through it:
  `get_plugin("xgboost__ab12cd")` returns the `xgboost` plugin. A single-model run (no `__`) is
  unaffected. This is the one blocker from the investigation, fixed in one place.
- `plugin_descriptor` / catalogue are unchanged (grid points are not registered plugins; they are
  configured specs).

### C2. Parameter-space declaration on the plugin contract (`ml_engine/models/base.py`)

- Add a `parameter_space(self) -> dict[str, ParameterSpec]` method to `ModelPlugin`, default
  empty, overridden per plugin. `ParameterSpec` (new versioned contract) declares, per searchable
  parameter: `type` (`int`/`float`/`bool`/`str`), optional `min`/`max` for numerics, optional
  `choices` for categoricals. This is the schema Req 1.4 validates against; it does not change how
  params reach the estimator (still `resolve_params` → `build`).
- Each built-in plugin (linear, random_forest, xgboost) declares the handful of parameters that
  are sensible to search (e.g. xgboost: `max_depth` int ≥ 1, `learning_rate` float in (0, 1],
  `n_estimators` int ≥ 1, `reg_alpha`/`reg_lambda` float ≥ 0, `subsample`/`colsample_bytree` float
  in (0, 1]). A parameter not in the space is rejected with its name (Req 1.4). Values outside a
  declared bound are rejected.
- Inherited-file change (additive): `base.py` gains the method with an empty default; each plugin
  gains its space. Manifest update + justification.

### C3. Grid contracts (`ml_engine/contracts/grid.py`, NEW)

- `ParameterGrid` = `dict[str, dict[str, list[scalar]]]` wrapped in a versioned model:
  model name → parameter name → candidate values.
- `GridPoint`: `model` (base), `point_id` (`<base>__<hash8>`), `params` (the resolved combination),
  `display_name` (`<base>#<hash8>`).
- `GridPlan`: the full list of `GridPoint`s, `fold_count` (K or 1), `fits_per_point` (K+1 or 1),
  `total_fits`, and the per-model parameter sets searched — the object shown for approval (Req 3)
  and recorded in the audit trail (Req 3.4).

### C4. Grid expansion and guards (`ExperimentService`, + a pure `ml_engine` expander)

- A pure function `expand_grid(parameter_grid, base_specs, plugins) -> list[GridPoint]` in
  `ml_engine` (no IO): for each model with a grid entry, cross-product its parameter value lists
  into `GridPoint`s; a model with no grid entry stays a single default point (Req 1.5). Property 2
  tests it against the literal cross-product.
- `ExperimentService._resolve_models` (and a new `_expand_grid`): when `request.parameter_grid` is
  present, validate it (every model is in the run; every parameter is in the plugin's
  `parameter_space`; reject the non-parameter keys `features`/`split`/`outlier`/`imputation`,
  Req 1.3; validate each value, Req 1.4), expand to `GridPoint`s, and build one `ModelSpec` per
  point: `ModelSpec(name=point.point_id, params=point.params)`. These replace the plain specs in
  `config.models`.
- The **fit-count guard is already correct**: `_resolve_split` passes `model_count=len(models)` to
  `resolve_cv_folds`, and `len(models)` now equals the grid-point count, so
  `points × (K+1) > cv_max_fits` is refused with the counts (Req 4.1, 4.2) before anything is
  archived or submitted (Req 4.5). The concurrency ceiling (`max_models_per_run`) and the budget
  guard likewise bound the expanded list unchanged (Req 4.3, 4.4).
- The `GridPlan` is returned so the UI/MCP can show it; approval is the act of submitting training
  with the grid (the same propose/approve shape as every other mutation — the LLM may suggest the
  grid, the user submits it, Req 3.2, 3.3).

### C5. Training, evaluation, versions — reused unchanged

- **Orchestration:** one training job per grid point (the orchestrator already fans out over
  `config.enabled_models_specs()`); names flow to `models/<point_id>/`, so no point overwrites
  another (Req 5.2). A point's failure is isolated exactly as a model's is today (Req 5.3).
- **Training stage:** `run_training` / `run_cv_training` look up the spec by name and call
  `get_plugin(model_name)` — now suffix-aware (C1) — then `resolve_params(spec.params)`; the
  per-fold preprocessing is the one fitted pipeline, identical for every point (Req 2.1, 2.2).
- **Evaluation:** `collect_model_results` + `build_comparison` + the CV `extend_comparison`/
  `rank_cv` rank every point by Best_By_CV_Mean, already (Req 6.1, 6.2); nothing parses `#`/`__`.
  The instability/outlier warnings apply per point (Req 6.4). No auto-promotion exists (Req 6.3).
- **Model versions:** each point's `models/<point_id>/metadata.json` is listed by
  `ModelVersionService` as `current:<point_id>`, independently loadable and validatable (Req 5.4,
  5.5). The version record carries the grid's parameters (they are in `ModelMetadata.hyperparameters`)
  and the `display_name` so results trace to the Parameter_Grid (Req 5.5).

### C6. Surfacing (API/MCP, optional in this spec)

- `ParameterGrid` is an optional field on `TrainingRequest`; absent reproduces today (Req 7.2).
- The `GridPlan` is returned by `start_training` (echoed like the resolved CV decision) so the
  control-UI and MCP can show points and the fit count before approval. A dedicated grid view is a
  UI concern (open question 4); the comparison already lists every point.

## Data Models

- `ParameterSpec` (`ml_engine/contracts/grid.py`, NEW) — per-parameter type + bounds/choices.
- `ParameterGrid`, `GridPoint`, `GridPlan` (same module, NEW) — versioned Pydantic.
- `TrainingRequest.parameter_grid: ParameterGrid | None` (new optional field, `backend/schemas.py`).
- `ModelSpec` unchanged — grid points reuse it (name + params).
- `ModelMetadata` unchanged — a grid point records its params in `hyperparameters` already; the
  `display_name` carries the readable `<base>#<hash>`.

No existing artifact path changes; grid points live under `models/<point_id>/` via the existing
`model_metadata`/`model_artifact` methods. Layout is unchanged (Req 7.3 is satisfied without a new
path).

## Correctness Properties

### Property 1: grid expansion is the cross-product
`expand_grid` produces exactly the cross-product of each model's per-parameter value lists; a model
with no grid entry yields one default point.
**Validates: Requirements 1.2, 1.5**

### Property 2: fit count is points × (K + 1) or points × 1
The reported `total_fits` equals the number of grid points times (K + 1) in CV_Mode, or times 1 in
Single_Split_Mode.
**Validates: Requirements 3.1**

### Property 3: non-parameter keys and invalid values are refused
A grid naming a non-parameter key (feature set, split, outlier, imputation) or an unknown parameter
or an out-of-space value is rejected before submission with the offending key/value.
**Validates: Requirements 1.3, 1.4**

### Property 4: the ceiling refuses, never trims
A grid whose total fits exceed `cv_max_fits` is refused with the requested and allowed counts, and
no job is submitted; the grid is never silently shrunk.
**Validates: Requirements 4.1, 4.2, 4.5**

### Property 5: points are isolated and additive
Each grid point is a distinct, loadable Model_Version; one point's failure leaves the others
intact; no point overwrites another's artifacts.
**Validates: Requirements 5.2, 5.3, 5.4**

### Property 6: absence reproduces today
A training request with no `parameter_grid` produces exactly today's one-spec-per-model run,
byte-compatible.
**Validates: Requirements 7.2, 8.5**

## Error Handling

- Grid validation failures (`_resolve_models`/`_expand_grid`) raise `ValidationError` with the
  offending key/value and a stable `reason`; nothing is archived or submitted.
- The fit-count/budget guards raise before `archive_previous_run` (they already do), so a refused
  grid leaves the previous run untouched.
- A single grid point's training failure writes `models/<point_id>/failure.json`; evaluation
  records it and ranks the rest (existing per-model isolation).

## Testing Strategy

Offline, with `InMemoryObjectStore` + `InlineOrchestrator` + mocked AWS (the CV/grid tests already
run this way).

- Property test: expansion equals the cross-product and the fit count equals points × (K+1) / ×1
  (Req 8.2).
- A grid over `cv_max_fits` is refused with the counts and submits no job; a grid varying a
  non-parameter key is rejected (Req 8.3).
- Every grid point becomes a distinct, loadable Model_Version; one point's failure leaves the
  others intact (Req 8.4).
- A request with no grid behaves exactly as today (Req 8.5).
- Parameter-space validation: unknown parameter and out-of-bound value rejected; a valid grid
  passes.

## What changes in inherited files (manifest)

| File | Change | Justification |
|---|---|---|
| `ml_engine/models/registry.py` | Add `base_model_name` and make `get_plugin` suffix-aware | A grid point's name `<base>__<hash>` must resolve to its base plugin; single-model names unaffected |
| `ml_engine/models/base.py` | Add `parameter_space()` (empty default) | The schema grid values are validated against (Req 1.4); does not change param resolution |
| `ml_engine/models/linear.py`, `random_forest.py`, `xgboost_plugin.py` | Declare each plugin's searchable `parameter_space` | The per-model searchable parameters and their bounds |

Each edit is additive and recorded in `tests/inherited_manifest.json` with a written justification,
regenerated via `scripts/inherited_manifest.py --write`. The `ml-factory-2` "no HPO" boundary is
amended (Req 7.5) to permit user-approved grid search (a doc note, like the stock-image banner).

## Open questions (resolved in design)

- **Run mode (Q1):** run under the experiment's mode (CV or single split); fit count scales
  accordingly.
- **Naming (Q2):** path/job id `<base>__<hash8>`; display `<base>#<hash8>`; full params in the
  version record.
- **Per-parameter cap (Q3):** none; `cv_max_fits` is the single guard.
- **Report (Q4):** the comparison already lists every point; a dedicated grid view is a UI concern.
- **Retention (Q5):** grid-point versions are ordinary Model_Versions, same retention.
