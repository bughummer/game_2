# Implementation Plan: Grid search over model parameters

## Overview

Python, following `design.md`. The core insight: a grid point is an ordinary `ModelSpec`, so most
of the pipeline is reused. Build the plumbing bottom-up — base-plugin lookup, parameter-space
declarations, grid contracts, the pure expander — then the service-layer expansion + guards + the
Grid_Plan, then surface it, then the offline tests. Every inherited-file change is additive and
followed by a manifest update. Commands: `.venv\Scripts\python -m pytest -q`,
`.venv\Scripts\python -m ruff check .`, `.venv\Scripts\python -m ruff format --check .`.

## Task Dependency Graph

```
1 grid id scheme + base-plugin lookup (inherited: registry)
└─▶ 2 parameter_space on the plugin contract (inherited: base + plugins)
     └─▶ 3 grid contracts + pure expander (+ PBT)
          └─▶ 4 service: expand, validate, Grid_Plan, guards; request field
               └─▶ 5 checkpoint
                    └─▶ 6 surface the Grid_Plan (start_training echo; API/MCP optional)
                         └─▶ 7 end-to-end + backward-compat tests
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

- [x] 1. Grid id scheme and base-plugin lookup (inherited; design C1)
  - `ml_engine/models/registry.py`: add `base_model_name(name) -> str` (substring before `__`,
    else the name) and make `get_plugin` resolve through it, so `get_plugin("xgboost__ab12cd")`
    returns the xgboost plugin; single-model names are unchanged
  - Add a helper to build a grid-point id: `grid_point_id(base, params) -> str` = `f"{base}__{hash8}"`
    with `hash8` = first 8 hex of sha256 over the sorted params (stable, alnum, job-name-safe)
  - Manifest update + justification for `registry.py`; regenerate
  - Tests: `get_plugin` resolves a suffixed name; `base_model_name` round-trips; ids are stable and
    distinct for distinct params and sanitise to unique SageMaker job names
  - _Requirements: 5.2_

- [x] 2. Parameter-space declaration on the plugin contract (inherited; design C2)
  - `ml_engine/contracts/grid.py` (NEW): `ParameterSpec` (type, min/max, choices)
  - `ml_engine/models/base.py`: add `parameter_space(self) -> dict[str, ParameterSpec]`, empty
    default; a `validate_grid_value(name, value)` helper on the plugin that checks a value against
    the space and raises a structured error naming the parameter/value
  - Declare `parameter_space` on `linear.py`, `random_forest.py`, `xgboost_plugin.py` (the handful
    of searchable params per the design, with bounds)
  - Manifest updates + justifications for base.py and the three plugins; regenerate
  - Tests: each plugin exposes a non-empty space; an unknown parameter and an out-of-bound value
    are rejected; a valid value passes
  - _Requirements: 1.1, 1.4_

- [x] 3. Grid contracts and the pure expander (design C3, C4)
  - `ml_engine/contracts/grid.py`: `ParameterGrid`, `GridPoint`, `GridPlan` (versioned)
  - `ml_engine/models/grid.py` (NEW) `expand_grid(parameter_grid, base_names, plugins) -> list[GridPoint]`:
    cross-product per model; a model with no grid entry yields one default point; pure, no IO
  - Hypothesis property test (Property 1/2): expansion equals the cross-product; a model without a
    grid entry yields one point
  - _Requirements: 1.2, 1.5, 7.1_

- [x] 4. Service: expand, validate, Grid_Plan, guards; request field (design C4, C6)
  - `backend/schemas.py`: `TrainingRequest.parameter_grid: ParameterGrid | None = None` (optional)
  - `ExperimentService`: when `parameter_grid` is present, validate it (model in run; parameter in
    the plugin's space; reject non-parameter keys naming the offender; validate each value), expand
    to `GridPoint`s, and build one `ModelSpec(name=point.point_id, params=point.params)` per point;
    these become `config.models`. Compute the `GridPlan` (points, fits_per_point, total_fits)
  - The fit-count guard already consumes `len(models)` in `_resolve_split`; confirm the grid-point
    count flows in so `points × (K+1) > cv_max_fits` is refused with the counts before archiving
  - Freeze the grid provenance: the `GridPlan` is recorded in the audit trail on start_training
  - Tests: non-parameter key rejected (Req 1.3); over-ceiling grid refused with counts and submits
    no job (Req 4.1, 4.2, 4.5); a model without a grid entry trains once (Req 1.5); the frozen
    config carries one ModelSpec per point
  - _Requirements: 1.3, 1.6, 2.1, 2.2, 2.3, 3.1, 3.2, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5, 7.2_

- [x] 5. Checkpoint - Ensure all tests pass
  - pytest + ruff + manifest; ask the user if questions arise
  - Result: 1450 passed, 1 skipped; ruff clean; manifest regenerated (tasks 1-4)

- [x] 6. Surface the Grid_Plan (design C6)
  - `start_training` (MCP tool + control-ui route) gains an optional `parameter_grid` and echoes
    `grid_points` (count of expanded points, read from the frozen config) alongside the resolved
    evaluation decision; the LLM may suggest a grid, the user submits it (Req 3.3)
  - `start_training` (service → MCP tool + control-ui route) echoes the `GridPlan` alongside the
    resolved evaluation decision, so the UI/MCP can show the points and the fit count; the LLM may
    suggest a grid but the user submits it (Req 3.3)
  - Tests (in-process client / route): the echoed plan lists the points and total fit count
  - _Requirements: 3.1, 3.2, 3.3, 6.2_

- [x] 7. End-to-end and backward compatibility (design Testing Strategy)
  - Covered in `tests/backend/test_grid_search.py`: single-split and CV grid runs; every point a
    distinct loadable version ranked in the comparison; a property test that expansion equals the
    cross-product; a per-point failure leaves the others intact; no-grid is byte-compatible. The
    ml-factory-2 "no HPO" boundary is amended (user-approved grid search is in scope).
  - End-to-end offline grid run (InlineOrchestrator): a small grid (e.g. xgboost max_depth∈{3,5},
    reg_lambda∈{1,2} → 4 points) under single split and under CV; assert each point is a distinct,
    loadable Model_Version, all points ranked by Best_By_CV_Mean, one injected point failure leaves
    the others intact (Req 5.3, 5.4, 6.1)
  - Backward compatibility: a request with no grid is byte-compatible with today (Req 8.5)
  - Amend the ml-factory-2 "no HPO" boundary (a doc note; Req 7.5)
  - _Requirements: 5.3, 5.4, 6.1, 6.3, 8.4, 8.5, 7.5_

- [x] 8. Final checkpoint - Ensure all tests pass
  - Run `.venv\Scripts\python -m pytest -q`, `.venv\Scripts\python -m ruff check .`,
    `.venv\Scripts\python -m ruff format --check .`; confirm `tests/test_inherited_manifest.py`
    passes with the regenerated manifest; ensure all pass; ask the user if questions arise
  - Result: 1453 passed, 1 skipped; ruff check + format clean; manifest + architecture-boundary
    tests pass

## Notes

- A grid point is a `ModelSpec`; orchestration, training (single + CV), evaluation/ranking and the
  model-version store are reused with no change. The only engine edits are the base-plugin lookup
  (task 1), the parameter-space declaration (task 2), and the grid contracts/expander (task 3).
- The fit-count, concurrency and budget guards already bound a list of models; the grid feeds them
  the expanded point count, so a large grid is refused by the existing guards.
- Property tests use Hypothesis (`max_examples=100`, `deadline=None`, `derandomize=True`), tagged
  `# Feature: grid-search, Property N`.
