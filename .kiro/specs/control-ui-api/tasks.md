# Implementation Plan: Control UI and HTTP API

## Overview

Python, following `design.md`. Build the thin HTTP API over the existing services first (it needs
no new engine work), then the Model_Version_Store (read model over existing per-run storage), then
validation-on-a-labeled-dataset (a new side job generalising holdout), then the UI, then the
end-to-end checkpoint. The API and MCP always call the same service methods. Commands:
`.venv\Scripts\python -m pytest -q`, `.venv\Scripts\python -m ruff check .`,
`.venv\Scripts\python -m ruff format --check .`. Every inherited-file change updates
`scripts/inherited_manifest.py` and regenerates `tests/inherited_manifest.json` in the same task.

## Task Dependency Graph

```
1 REST skeleton (app mount, error map, deps, health)
└─▶ 2 experiment + dataset + EDA/feature routes
     └─▶ 3 training + analysis + report/download routes
          └─▶ 4 checkpoint
               ├─▶ 5 ModelVersion contract + store service + routes
               └─▶ 6 Validation contracts + side job + service + routes
                    └─▶ 7 checkpoint
                         └─▶ 8 UI (thin SPA)
                              └─▶ 9 end-to-end + final checkpoint
```

```json
{
  "waves": [
    { "wave": 1, "tasks": ["1"], "depends_on": [] },
    { "wave": 2, "tasks": ["2"], "depends_on": ["1"] },
    { "wave": 3, "tasks": ["3"], "depends_on": ["2"] },
    { "wave": 4, "tasks": ["4"], "depends_on": ["3"] },
    { "wave": 5, "tasks": ["5", "6"], "depends_on": ["4"] },
    { "wave": 6, "tasks": ["7"], "depends_on": ["5", "6"] },
    { "wave": 7, "tasks": ["8"], "depends_on": ["7"] },
    { "wave": 8, "tasks": ["9"], "depends_on": ["8"] }
  ]
}
```

## Tasks

- [x] 1. REST skeleton: app, error mapping, principal, health (design C1, C2)
  - Built on **Starlette routes** (not FastAPI): FastAPI is not installed and the HTTP app already
    uses FastMCP's Starlette `custom_route` for `/health`, so the API attaches its routes to the
    same FastMCP app via `mcp.custom_route(f"/api{suffix}", ...)`. No new dependency; `app.run()`
    and the SSE tests are unchanged. (Design C1 named FastAPI as optional sugar; Starlette is the
    lower-risk peer here.)
  - `backend/api/errors.py`: `STATUS_BY_CODE` map (C2), `error_response`/`generic_error_response`,
    and a `safe(handler)` wrapper (the HTTP sibling of the MCP's `_safe`) — AppError -> its status,
    anything else -> 500 with no trace
  - `backend/api/deps.py`: `principal(container, request)` via `authorizer.principal` (a
    `X-ML-Factory-User` header is only a claim; dev identity wins), `require_owner`, `audit_mutation`
  - `backend/api/app.py`: `register_api_routes(mcp, container)` attaching routes under `/api`;
    `_collect_routes()` is the extension point for later resource modules; task 1 registers
    `/api/health`. Wired into `mcp_server/app.py::build_app`
  - Tests in `tests/mcp/test_api_skeleton.py`: `/api/health` mounted beside `/health` and `/sse`;
    error map per `AppError`; `safe` hides unexpected errors (no leak); principal ignores the claim
    header
  - _Requirements: 1.3, 11.2, 11.4, 11.5_

- [x] 2. Experiment, dataset, EDA and feature routes (design C3)
  - Routes wrapping `ExperimentService.create/get/list/delete`, `LeakageService.set/get_business_context`,
    `DatasetService.create_upload_ticket/register/list_uploads` + bounded preview,
    `ExperimentService.eda/leakage/feature_review` (inferred problem type + override),
    `FeatureProposalService.generate/define/approve` (+ Bedrock-off -> `FEATURE_DISABLED`),
    `LeakageService` investigation + decisions, `ExperimentService.save_features`
  - Every mutation: `principal` -> `require_owner` -> service -> `audit.record`
  - Tests: each route success; ownership 403; not-ready/invalid-state mapping; previews bounded;
    both the API and MCP call the same service (assert by behaviour)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.4, 5.5_

- [x] 3. Training, analysis, report and download routes (design C3)
  - Routes wrapping `AnalysisService.preflight/execution_status/permutation_importance/ablation`,
    `ExperimentService.start_training/training_status/stop_training/comparison`,
    `LeakageService.build_post_training_phase`, `ReasoningService.interpret`,
    `ReportingService` report generate/get, `DatasetService.create_download_url` (experiment-scoped)
  - Confirm all guards run in the service (the API adds/bypasses none); start_training returns an
    execution id immediately
  - Tests: start/stop/status; preflight surfaced; guard violation mapped from the service; download
    link scoped to the experiment and refused for a foreign URI
  - _Requirements: 5.3, 6.1, 6.2, 6.3, 6.4, 6.5, 9.4, 10.1, 10.2_

- [x] 4. Checkpoint - Ensure all tests pass
  - pytest + ruff; ask the user if questions arise
  - Result: 1387 passed, 1 skipped; ruff check + format clean (tasks 1-3)

- [x] 5. Model_Version_Store: contract, service, routes (design C4)
  - `ml_engine/contracts/model_version.py` (NEW): `ModelVersion` (versioned) with `version_id`
    (`<run_id>:<model>` / `current:<model>`), metadata, `source`, `run_id`, `artifact_ref`
    (model + metadata + preprocessor URIs for independent loadability)
  - `backend/services/model_versions.py` (NEW) `ModelVersionService`: `list_versions(experiment_id)`
    enumerating the current run's `models/*/metadata.json` and every `runs/*/models/*/metadata.json`,
    newest run first; `get_version(experiment_id, version_id)` resolving the artifact refs;
    optional `model_versions/index.json` cache (one writer)
  - Layout: add `model_version_index`; register the new contract/service in the manifest if an
    inherited file is touched (layout.py) and regenerate
  - Routes: list versions, get one version
  - Tests: lists across current + archived runs, newest first; retrain adds versions without losing
    earlier ones; each version independently loadable; a pre-feature experiment lists its models
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 6. Validation on a labeled dataset: contracts, side job, service, routes (design C5)
  - `ml_engine/contracts/validation.py` (NEW): `ValidationRequest`, `ValidationModelResult`,
    `ValidationReport` reusing `RowOverlap`/`MetricSet`/`ThresholdAnalysis`/`ThresholdPoint`
  - Inherited (manifest): `ml_engine/contracts/execution.py` add `WorkflowKind.VALIDATION` +
    `SIDE_WORKFLOWS`; `backend/orchestration/sagemaker.py` add `STAGE_MODULES["validation"]` and
    `start_validation(record, validation_id)` via `_start_side_job(..., image_uri=training_image)`;
    `ml_engine/io/layout.py` add `validation_request/validation_report/validation_failure/
    validation_input_dir`
  - `jobs/validation/main.py` (NEW) `run_validation(...)`: from the holdout job body, iterate the
    requested Model_Versions (resolve each version's model + preprocessor incl. archived runs),
    drop the label from input but keep it for metrics, compute `evaluate(...)` on all/unseen rows,
    `RowOverlap`, and (binary + labels) `threshold_analysis(...)` on the validation labels; labels
    absent -> predictions + distribution + "no metrics"; write `ValidationReport`
  - `backend/services/validation.py` (NEW) `ValidationService`: `start_validation(experiment_id, *,
    dataset_uri, version_ids=None, file_format=None, user=None)` (empty = all versions) and a read
    method; validate + pin the dataset; one job loops the versions; refuse re-fit
  - Fold holdout in: `start_holdout_evaluation` becomes a thin alias calling validation with the
    current run's model versions
  - Routes: start validation (one/all), get validation result, compare across versions
  - Manifest update + regenerate for the three inherited edits
  - Tests: label dropped for prediction, kept for metrics (Property 4); threshold analysis on the
    validation labels (Property 5); all-versions comparison; labels-absent path; overlap/unseen
    split; validation is a side job and does not change training state
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 9.1, 9.2, 9.3, 9.5, 9.6_

- [x] 7. Checkpoint - Ensure all tests pass
  - pytest + ruff + `tests/test_inherited_manifest.py`; ask the user if questions arise
  - Result: 1424 passed, 1 skipped; ruff clean; manifest + architecture-boundary tests pass
    (tasks 5-6: Model_Version store and validation side job)

- [x] 8. The UI (design C6) — thin static client (option 1)
  - A dependency-free static client (`backend/api/ui/{index.html,styles.css,app.js}`) served by the
    API: shell at `/`, assets at `/ui/<name>` via `backend/api/static.py`; registered in
    `register_api_routes`. No build step, no new toolchain; `pyproject` ships the assets as
    package-data
  - Screens for the full 7-step flow (create → data → EDA → features → train → versions → validate);
    calls only `/api`; deterministic values render monospace/cobalt, LLM output in a violet
    "suggested" block (the design law); previews bounded; no metric computed client-side
  - Tests in `tests/mcp/test_api_ui.py`: shell + assets served with correct content types; unknown
    asset refused (404)
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 9. End-to-end and final checkpoint (design Testing Strategy)
  - A full offline flow test with the in-process container (InlineOrchestrator): create → EDA →
    approve features → train → list Model_Versions → validate on a labeled dataset → post-prediction
    analysis incl. threshold analysis; assert label handling and threshold-on-validation
  - Implemented in `tests/mcp/test_api_end_to_end.py`
  - Result: 1425 passed, 1 skipped; ruff check + format clean; inherited manifest + architecture
    boundary tests pass
  - _Requirements: 13.1, 13.2, 13.3_

## Notes

- Tasks 1–4 wrap existing services only — no engine change, no manifest churn. The inherited edits
  are confined to task 6 (validation side job) and the small layout additions in tasks 5–6.
- Validation reuses the holdout job's label/overlap/threshold logic rather than the scoring job's
  (scoring deliberately ignores labels).
- Auth stays the MCP's deferred boundary (flagged dev identity) throughout.
