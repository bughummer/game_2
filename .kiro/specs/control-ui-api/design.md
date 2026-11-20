# Design: Control UI and HTTP API

## Overview

Add a web UI and the HTTP API behind it as a **second front-end**, a peer of the existing MCP over
the one service layer (`backend/services/*`). The API holds no business logic — it is a thin HTTP
wrapper that resolves the caller, enforces ownership, calls a service function, maps `AppError` to
an HTTP status, and audits mutations, exactly as the MCP's `_safe`/`principal`/`record` helpers do.

On top of surfacing the existing flow, this spec adds two capabilities the flow needs:

- a **Model_Version_Store** — a durable, listable set of every model produced in an experiment
  (each algorithm, each retrain, each future grid-search point), expressed additively over the
  per-run storage that already exists (`runs/<run_id>/` + `ArchivedRun`), so nothing is overwritten;
- **validation of a Model_Version on an uploaded labeled dataset**, which generalises the existing
  holdout-evaluation job (labels kept for metrics, overlap reported, threshold analysis on the
  validation labels) to target a chosen version (or all of them).

## Architecture

```
            ┌──────────── one control-plane process (one Fargate service) ───────────┐
  Browser ──┤  REST API (Starlette/FastAPI routes)  ─┐                                │
            │                                        ├─▶ AppContainer ─▶ services ─▶ … │
  LibreChat ┤  MCP server (FastMCP, SSE + /health)  ─┘   (one shared instance)         │
            └────────────────────────────────────────────────────────────────────────┘
                                   services ─▶ orchestrator ─▶ SageMaker jobs ─▶ S3
```

- **One process, one container, one `AppContainer`.** `mcp_server/app.py::build_app` already builds
  a FastMCP whose `http_app()` is a Starlette app with a `/health` custom route. The REST API is
  added as Starlette routes on the same ASGI app (or a sibling app mounted under `/api` in a parent
  ASGI app), sharing the same `AppContainer` the MCP uses. No new process, no duplicated wiring.
- **The REST layer is the MCP's twin.** For every capability, the REST handler and the MCP tool call
  the *same* service method. A behaviour change is made once, in the service.
- **Platform invariants unchanged:** SageMaker-only execution, S3-only artifacts, every number from
  the engine or tracking, ownership + audit on mutations, the LLM proposes/narrates but computes no
  number and applies nothing without approval (B12), the default suite runs offline.

## Components and Interfaces

### C1. REST application (`backend/api/`, NEW)

A small package: `app.py` (route registration), `errors.py` (AppError→HTTP mapping), `deps.py`
(principal resolution from the request), `routes/*.py` (one module per resource).

- **Framework.** Starlette routes (FastAPI optional sugar on top). The error module is already
  framework-free, so the mapping is a thin translator. Decision: FastAPI, because it gives request
  validation and OpenAPI for free and runs as a Starlette app that mounts cleanly beside FastMCP.
- **Mounting.** `build_app(container)` gains the REST app: either `@mcp.custom_route` per route, or
  a parent `Starlette(routes=[Mount("/api", api_app), Mount("/", mcp.http_app(...))])`. Decision:
  a parent ASGI app mounting the MCP ASGI app and the FastAPI app, so `/api/...`, `/sse`, `/health`
  coexist and the health route stays shared.
- **Request→service pattern (every mutation):**
  ```
  who = container.authorizer.principal(user_from_request)
  container.authorizer.require_owner(who, experiment_id)   # ownership-scoped routes
  result = service.method(...)                              # the only place logic runs
  container.audit.record(artifact_prefix=..., experiment_id=..., action=..., actor=who.user, ...)
  return json(result)
  ```
  Read routes skip `require_owner`/`record` (reads are not audited, matching the MCP).

### C2. Error mapping (`backend/api/errors.py`, NEW)

A single table from the stable `AppError.code` to an HTTP status, mirroring the MCP's `_safe`
(which wraps the same codes into `ToolError`):

| code | status |
|---|---|
| `VALIDATION_ERROR` | 400 |
| `NOT_AUTHORIZED` | 403 |
| `FEATURE_DISABLED` | 403 |
| `NOT_FOUND` | 404 |
| `INVALID_STATE` | 409 |
| `ARTIFACT_NOT_READY` | 409 |
| `UPSTREAM_ERROR` | 502 |
| `INTERNAL_ERROR` (and anything uncaught) | 500 |

The body is `AppError.as_dict()` (`{code, message, details}`); non-`AppError` exceptions never leak
a trace (500 with a generic body), as the MCP does.

### C3. Route surface (all wrapping existing services)

One module per resource; every handler is a thin call. Representative set (verbs omitted for
brevity, all under `/api`):

- **experiments**: create / get / list / delete; set + get business context —
  `ExperimentService.create/get/list/delete`, `LeakageService.set/get_business_context`.
- **datasets**: upload-ticket / register / list uploads / preview —
  `DatasetService.create_upload_ticket/register/list_uploads`; preview reuses the bounded preview.
- **eda / features**: eda, leakage, feature-review, inferred problem type —
  `ExperimentService.eda/leakage/feature_review`; propose / define / approve features and leakage
  decisions — `FeatureProposalService`, `LeakageService`; `save_features`.
- **training**: preflight, start_training, training_status, execution_status, stop_training —
  `AnalysisService.preflight/execution_status`, `ExperimentService.start_training/training_status/
  stop_training`.
- **model versions**: list versions; get one version — new `ModelVersionService` (C4).
- **validation**: start validation (one version / all), get validation result, compare —
  new `ValidationService` (C5).
- **analysis**: comparison, permutation, ablation, post-training leakage, interpret —
  `ExperimentService.comparison`, `AnalysisService.permutation_importance/ablation`,
  `LeakageService.build_post_training_phase`, `ReasoningService.interpret`.
- **reports / downloads**: generate/get report, download link — `ReportingService`,
  `DatasetService.create_download_url`.

### C4. Model_Version_Store (`backend/services/model_versions.py`, NEW service; additive layout)

A Model_Version is a stable, listable handle to one trained model within a run, resolvable to its
artifact + preprocessing + metadata. The per-run storage already exists; this adds an index, not a
new copy of the bytes.

- **Contract** `ModelVersion` (versioned Pydantic, `ml_engine/contracts/model_version.py`, NEW):
  `version_id` (stable: `<run_id>:<model_name>`, or `current:<model_name>` for the live run),
  `experiment_id`, `model_name`, `display_name`, `algorithm`/`library`, `hyperparameters`,
  `problem_type`, `primary_metric`, `primary_score`, `metrics`, `created_at`, `source`
  (`training` | `retrain` | `grid_point`), `run_id` (None for the current run), and
  `artifact_ref` — the resolved URIs of the model, its metadata, and the preprocessing strategy it
  needs (so it is independently loadable, Req 7.5).
- **Store** = a thin read model over what already exists, so nothing new is written at training
  time in Stage 1 of this spec:
  - the **current run's** models: `models/<name>/metadata.json` (+ `model.joblib`, the preprocessor
    for `plugin.preprocessing_strategy`);
  - **archived runs**: `runs/<run_id>/models/<name>/metadata.json` (+ the archived model and
    preprocessor under the same `runs/<run_id>/` prefix).
  `list_versions(experiment_id)` enumerates the current run's `models/*/metadata.json` and every
  `runs/*/models/*/metadata.json`, builds a `ModelVersion` per model, newest run first. A
  `model_version_index.json` (one writer: the version service, written on demand/first list) MAY
  cache the enumeration; the source of truth stays the per-model metadata.
- **Backward compatibility (Req 7.6):** an experiment trained before this spec has
  `models/*/metadata.json` and possibly `runs/*/...`; `list_versions` presents them as
  Model_Versions with no re-training.
- **No overwrite (Req 7.2):** `archive_previous_run` already copies the prior run to
  `runs/<run_id>/` before a retrain and writes an `ArchivedRun` manifest. Expressed as versions:
  retrain *adds* the archived run's versions to the store; the new run adds its own. Nothing is
  deleted beyond the copy-then-delete that already moves bytes into the archive.
- **Layout additions (`ml_engine/io/layout.py`):** `model_version_index` →
  `model_versions/index.json` (new, optional cache); the per-model and per-run paths already exist.
  No existing path changes.

### C5. Validation on a labeled dataset (`backend/services/validation.py` + `jobs/validation/`, NEW)

Generalises holdout evaluation to target a chosen Model_Version (or all), keeping the holdout job's
label handling and overlap logic, and adding validation-set threshold analysis.

- **New side workflow** (following the item-9 pattern): `WorkflowKind.VALIDATION = "validation"`,
  added to `SIDE_WORKFLOWS`; `STAGE_MODULES["validation"] = "jobs.validation.main"`;
  `SageMakerOrchestrator.start_validation(record, validation_id)` via `_start_side_job(...,
  image_uri=self.training_image(experiment_id))` so models load in their training environment. A
  validation never changes the experiment's training state (Req 8.8).
- **Request/report contracts** (`ml_engine/contracts/validation.py`, NEW):
  - `ValidationRequest`: `validation_id`, `experiment_id`, `dataset_uri`, `file_format`,
    `version_ids` (empty = all versions), `requested_at/by`, `execution_id`.
  - `ValidationReport`: per-version `ValidationModelResult` reusing the holdout shapes —
    `RowOverlap`, metrics on all rows and on unseen rows (`ml_engine.evaluation.evaluate`), and
    (binary) a fresh `ThresholdAnalysis` computed on the validation labels
    (`ml_engine.evaluation.thresholds.threshold_analysis`), plus the default and F1-max points.
    Labels absent → predictions and their distribution only, with a "no metrics" note (Req 8.6).
- **Job** `jobs/validation/main.py::run_validation(store, layout, *, experiment_id, validation_id)`:
  starts from `run_holdout_evaluation`'s body; differences: it iterates the **requested
  Model_Versions** (resolving each version's model + preprocessor via the `ModelVersion.artifact_ref`,
  including archived runs), and it computes `threshold_analysis` on the validation labels (Req 9.2,
  9.3) rather than only re-applying the training-chosen threshold. Reuses `_derive`, `_overlap`,
  `_seen_and_training_frames`, `evaluate`, `point_at`.
- **"All versions" (Open question 4):** one SageMaker job loops the versions (keeps job count/cost
  down, mirrors holdout), writing one `ValidationReport` with a result per version (Req 8.7).
- **Folding in holdout (Open question 6):** holdout evaluation becomes the "validate the current
  run's models" special case. The existing `start_holdout_evaluation` stays as a thin alias that
  calls validation with `version_ids = current run's models`, so nothing breaks; new callers use
  the version-targeted API.
- **No re-fit (Req 8.3):** the job only loads and applies; it fits nothing (same as holdout).

### C6. The UI (`ui/`, NEW; thin SPA)

- A single-page app (framework chosen in implementation; kept thin) that calls only the API.
- Screens map 1:1 to the flow (Req 12.1): create experiment → upload/point S3 → EDA + problem type
  + leakage → feature proposals (approve) → choose algorithms + preflight + train → watch status →
  post-training analysis → model-version list → start validation on a labeled dataset →
  post-prediction analysis incl. threshold curve.
- **LLM output is visibly distinct** from deterministic results (Req 12.3); previews are bounded
  (Req 12.4); if Bedrock is off, feature proposals/interpretation degrade gracefully and the rest
  of the flow works (Req 12.5). The UI computes no metric and holds no logic (Req 12.2).

## Data Models

New versioned Pydantic contracts (paths only in `ml_engine/io/layout.py`):

- `ModelVersion` (`ml_engine/contracts/model_version.py`) — the listable version handle (C4).
- `ValidationRequest`, `ValidationModelResult`, `ValidationReport`
  (`ml_engine/contracts/validation.py`) — reusing `RowOverlap`, `MetricSet`, `ThresholdAnalysis`,
  `ThresholdPoint` from the existing contracts (C5).

Reused unchanged: `ModelMetadata`, `ArchivedRun`, `HoldoutEvaluationReport` shapes, `MetricSet`,
`ThresholdAnalysis`. Layout additions are purely new paths (`model_versions/index.json`,
`validation/<id>/request.json`, `.../validation.json`, `.../failure.json`, input dir); no existing
path changes, so existing artifacts stay byte-identical.

## Correctness Properties

### Property 1: one implementation behind two front-ends
For every capability, the REST handler and the MCP tool invoke the same `Service_Layer` function;
the API adds no ML/business logic.
**Validates: Requirements 1.1, 1.2, 1.6**

### Property 2: ownership and audit on every mutation
Every state-changing REST route resolves a `Principal`, calls `require_owner`, and records an audit
event; a non-owner is refused with 403.
**Validates: Requirements 1.4, 1.5, 11.1, 11.3**

### Property 3: error mapping without leakage
Every `AppError` maps to its fixed HTTP status with `{code, message, details}`; no other exception
leaks a trace.
**Validates: Requirements 1.3**

### Property 4: labels dropped for prediction, kept for metrics
When a Validation_Dataset carries the Label_Column, validation excludes it from the model input and
uses it only for metrics and threshold analysis; no model or transform is re-fit on it.
**Validates: Requirements 8.2, 8.3, 8.4**

### Property 5: threshold analysis on validation labels
For a binary Model_Version with labels present, the threshold analysis is computed on the
Validation_Dataset's own labels, not reused from training.
**Validates: Requirements 9.2, 9.3**

### Property 6: versions are additive and loadable
A retrain adds Model_Versions without deleting an earlier version's artifacts; each Model_Version is
independently loadable from its record alone; pre-feature experiments list their models as versions.
**Validates: Requirements 7.2, 7.5, 7.6**

## Error Handling

- Service errors surface as `AppError` → mapped status (C2); the API never constructs its own ML
  errors.
- A validation on an experiment with no trained models, or before EDA, returns `ARTIFACT_NOT_READY`
  / `INVALID_STATE` from the service unchanged.
- A failed validation job writes `validation/<id>/failure.json`; the read route reports `failed`
  with the message, like holdout/scoring.
- Bedrock-off: proposal/interpret routes return `FEATURE_DISABLED` (403); the rest of the flow is
  unaffected.

## Testing Strategy

Offline, mirroring `tests/mcp` (`InMemoryObjectStore`/moto S3, `FakeSageMaker`, in-process
`AppContainer`; `InlineOrchestrator` for the end-to-end flow where artifacts must really exist).

- Route tests per resource: success, ownership refusal (403), error mapping (404/409/400), bounded
  outputs.
- Model-version tests: list across current + archived runs; additivity after a retrain;
  independent loadability; a pre-feature experiment lists versions.
- Validation tests: label dropped for prediction and kept for metrics; threshold analysis on
  validation labels; all-versions comparison; labels-absent produces predictions + "no metrics";
  overlap/unseen split; runs as a side job and does not change training state.
- Full flow test: create → EDA → approve features → train → list versions → validate → analyze.

## What changes in inherited files (manifest)

| File | Change | Justification |
|---|---|---|
| `ml_engine/io/layout.py` | Add `model_version_index` and `validation_*` path methods; no existing path changes | Additive paths for the version index and the validation side job (Req 7.1, 8.1); defined only here per the layout rule |
| `ml_engine/contracts/execution.py` | Add `WorkflowKind.VALIDATION`; add to `SIDE_WORKFLOWS` | A new side workflow (validation) that never touches experiment state, same shape as holdout/scoring |
| `backend/orchestration/sagemaker.py` | Add `STAGE_MODULES["validation"]` and `start_validation` | Submit the validation side job, reusing the training image (same pattern as scoring) |

Each inherited edit is additive and recorded in `tests/inherited_manifest.json` with a written
justification, regenerated via `scripts/inherited_manifest.py --write`.

## Open questions (resolved in design)

- **Framework (Q1):** FastAPI in the same process/container as the MCP, mounted beside the FastMCP
  ASGI app over one `AppContainer`.
- **UI stack (Q2):** a thin SPA served as static assets; exact framework chosen at implementation;
  no logic in the client.
- **Validate all vs one (Q3):** both — the API takes `version_ids` (empty = all), one job loops.
- **One job or many (Q4):** one job loops the versions (cost/coherence; mirrors holdout).
- **Auth (Q5):** the MCP's deferred boundary — flagged dev identity until real auth is wired.
- **Holdout vs validation (Q6):** holdout evaluation is folded into validation as the
  "current-run models" case; the old entry stays as a thin alias.
