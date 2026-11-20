# Implementation Plan

## Overview

Build ML Factory 2.0 on SageMaker + S3 + MCP, inheriting the deterministic engine from
`../old_repo`. No local execution. The default suite runs offline (mocked AWS) and stays green
after every task; real AWS is opt-in via `ML_FACTORY_AWS_TESTS=1`.

## Task Dependency Graph

```
0 assessment (done)
└─▶ 1 scaffold + inherit engine
     └─▶ 2 S3 store
          ├─▶ 3 stage container + SageMaker orchestrator (boto3)
          │    └─▶ 4 tracking: status · Spot · cost · diagnostics
          │         ├─▶ 5 Step Functions
          │         └─▶ 8 reporting
          ├─▶ 6 backend port + MCP + ingestion   (integrates 3–4 as they land)
          └─▶ 7 business context + leakage + features
9 security/governance — increments inside every phase
```

```json
{
  "waves": [
    { "wave": 1, "tasks": ["0"], "depends_on": [] },
    { "wave": 2, "tasks": ["1"], "depends_on": ["0"] },
    { "wave": 3, "tasks": ["2"], "depends_on": ["1"] },
    { "wave": 4, "tasks": ["3", "7"], "depends_on": ["2"] },
    { "wave": 5, "tasks": ["4", "6"], "depends_on": ["3"] },
    { "wave": 6, "tasks": ["5", "8"], "depends_on": ["4"] },
    { "wave": 7, "tasks": ["9"], "depends_on": ["1", "2", "3", "4", "5", "6", "7", "8"] }
  ]
}
```

## Tasks

- [x] 0. Assessment of the reference repository
  - Verified seams, inherited test baseline (490 pass), absence of git history, plugin
    checkpoint constraints. See `assessment.md`.
  - _Requirements: 1_

- [x] 1. Scaffold the project and inherit the engine
  - [x] 1.1 Create `pyproject.toml` (Python ≥3.12, exact pins, extras: `jobs`, `models`,
    `dictionary`, `backend`, `dev` incl. `moto`, `pytest`, `ruff`), `Makefile`, `.gitignore`,
    `README.md`, git init. Local venv is Python 3.14 (the only interpreter installed).
    - _Requirements: 1_
  - [x] 1.2 Copy `ml_engine/` packages verbatim (all except `io/`); copy `io/layout.py`,
    `io/dataset.py`, and `io/object_store.py` minus `LocalObjectStore`.
    - _Requirements: 1.1_
  - [x] 1.3 Add `tests/support/in_memory_store.py` (`InMemoryObjectStore`, `s3://`-shaped URIs)
    and rewrite `tests/conftest.py` around it; drop queue/local fixtures; AWS env isolated;
    `aws` marker gated by `ML_FACTORY_AWS_TESTS=1`.
    - _Requirements: 1.2, 2.6, 3.7_
  - [x] 1.4 Copy `tests/ml_engine` and `tests/jobs`; replace the `file://`/`LocalObjectStore`
    tests in `test_io.py` with `s3://` tests; two job tests changed only in how they place a
    CSV in the store.
    - _Requirements: 1.2_
  - [x] 1.5 Inherited-file manifest (`tests/inherited_manifest.json`, 73 files, verified
    byte-identical to old_repo) + `scripts/inherited_manifest.py` + `tests/test_inherited_manifest.py`.
    - _Requirements: 1.5_
  - [x] 1.6 Architecture-boundary test: `ml_engine` imports no AWS SDK at all, no
    `fastapi`/`backend`/`jobs`/`mcp*`; jobs import boto3 only lazily in `runtime.py`; no
    local-store code may reappear.
    - _Requirements: 1.3, 1.4; Property 1_

- [x] 2. S3 as the only store
  - [x] 2.1 Rewrite `ml_engine/io/uri.py` for `s3://` (`parse_s3_uri`, `is_s3_uri`, `to_s3_uri`,
    `normalize_storage_root`, `InvalidS3UriError`); keep `join_uri`, `uri_matches_prefix`,
    `detect_format`; tests for valid/invalid URIs and boundary-safe prefix matching.
    - _Requirements: 2.3_
  - [x] 2.2 Write `ml_engine/io/s3.py` (`S3ObjectStore`, injected client, no SDK import,
    pagination, delimiter listing, boundary-safe prefixes, SSE-S3/KMS, `ObjectNotFoundError`
    mapping, non-missing errors propagate).
    - _Requirements: 2.1, 2.7_
  - [x] 2.3 Shared store contract suite parametrized over `InMemoryObjectStore` and `moto`
    `S3ObjectStore`; byte-identical contract round-trip; dataset loads; pagination; SSE.
    - _Requirements: 2.5, 2.6; Property 4_
  - [x] 2.4 Copy `jobs/`; `build_store()` returns `S3ObjectStore` (lazy boto3, standard retries,
    SSE from job env); `tests/jobs/test_s3_runtime.py` runs the profiling CLI end-to-end through
    the real `build_store` against moto.
    - _Requirements: 2.2, 3.2_

- [ ] 3. Stage container and SageMaker orchestrator (boto3)
  - [ ] 3.1 `container/Dockerfile` (python:3.14-slim matching the test env, pinned extras,
    engine + stages only, non-root `mlf` user, single-threaded BLAS) + `.dockerignore`; static
    checks in `tests/infrastructure/test_stage_image.py`.
    **Open:** `docker build` not yet run (Docker daemon unavailable on the dev machine); first
    real-AWS run must confirm a non-root container can sync `/opt/ml/checkpoints`.
    - _Requirements: 3.2_
  - [x] 3.2 Copy `backend/orchestration/base.py`; add `tests/support/inline_orchestrator.py`
    (same stage order as SageMaker, in-process).
    - _Requirements: 3.7_
  - [x] 3.3 `backend/config.py` (pydantic-settings, `ML_FACTORY_*`): S3/SageMaker/role/image/
    instances/Spot/budget/price-table settings; `problems()` reports what blocks submission;
    GPU instances rejected; `backend/aws/clients.py` ported (no static keys).
    - _Requirements: 2.2, 4, 15.3_
  - [x] 3.4 Pre-submission guards (`backend/orchestration/guards.py`, pure): CPU allow-list,
    Spot-only training unless opted in, concurrency, per-experiment and monthly budget against
    reserved upper-bound cost; unknown price or unverified price table → refused (fail closed).
    - _Requirements: 4.4, 4.5, 4.7; Property 11_
  - [x] 3.5 `backend/orchestration/sagemaker.py`: frozen config required; whole workflow guarded
    and reserved up front; deterministic job names; Processing for profiling/preparation/
    evaluation, one Spot Training job per model; `advance()` chains stages idempotently;
    `stop()`; opted-in On-Demand fallback on capacity failure only, once; never invents status.
    **Open:** `advance()` is called by the status monitor, which lands in task 4.2.
    - _Requirements: 3.1, 3.3, 3.4, 3.5, 4.1, 4.2, 4.6_
  - [x] 3.6 Checkpointing in `jobs/training/checkpointing.py` (no engine edit; Q3 not needed):
    XGBoost checkpoint callback + resume of remaining rounds; CatBoost native snapshot; sklearn
    models unwrapped; one `RESUMABLE_MODELS` list shared with the orchestrator.
    - _Requirements: 4.3_
  - [x] 3.7 Tests: fake SageMaker validating every request with botocore's API model; Spot,
    checkpoint, isolation, fallback, guards-before-AWS, idempotency, stop, unknown status;
    checkpoint identity/resume; in-process reproducibility within the documented tolerance.
    - _Requirements: 3.3, 3.5, 3.6, 4; Properties 5, 9, 11_

- [ ] 4. Status monitor and failure diagnostics
  Scope reduced by owner (2026-10-14): **no Spot/cost recording**. Spot utilization, savings and
  dollar costs are the MLOps team's domain (AWS Budgets / Cost Explorer / Canvas tooling), so
  requirement 7 is out of scope here and the price table stays optional (`enforce_budget`).
  - [x] 4.1 Status contracts `ml_engine/contracts/tracking.py` (`NormalizedStatus` incl. UNKNOWN,
    `JobObservation`, `FailureDiagnosis`, `FailureCategory`). No Spot/cost fields.
    - _Requirements: 6.2, 8.1_
  - [x] 4.2 Pure status normalization (`backend/tracking/status.py`): SageMaker primary +
    secondary status → user-facing status; UNKNOWN whenever evidence is insufficient; a Spot
    wait timeout reads as Failed, not a clean Stop.
    - _Requirements: 6.3, 6.4; Property 10_
  - [x] 4.3 Status monitor (`backend/tracking/monitor.py`): `on_event` (EventBridge) and `poll`
    (fallback) both re-observe the execution, persist a `JobObservation` per in-flight job, and
    call `advance()`; independent of chat; `SubmittedJob` now keeps raw + secondary status.
    - _Requirements: 6.1, 6.2_
  - [x] 4.4 Failure diagnostics (`backend/tracking/diagnostics.py`): ordered classifier over
    `FailureReason` + the stage's `failure.json` + bounded CloudWatch logs; S3 checked before
    the generic IAM rule; PII redaction (email, keys, ARNs, account ids, IPs, hex, long numbers)
    before any text could reach an LLM; log fetch never raises.
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [x] 4.5 Tests (fake SageMaker + fake Logs): status table incl. UNKNOWN; observations
    persisted; monitor drives a full run to completion; classifier categories and ordering;
    log bounding and redaction; diagnosis written once; no spend artifacts.
    - _Requirements: 6, 8; Property 10_

- [ ] 5. Step Functions orchestration
  - [ ] 5.1 ASL in `infra/stepfunctions/training.asl.json`: Prepare → Map(Train, per-item
    Catch) → Evaluate, `.sync` integrations, approval task token.
    - _Requirements: 3.5, 5.5_
  - [ ] 5.2 `backend/orchestration/step_functions.py` behind the same Protocol; container switch.
    - _Requirements: 3.1_
  - [ ] 5.3 Tests: ASL validation, start/describe via Stubber, per-model failure isolation.
    - _Requirements: 3.3, 3.5; Property 9_

- [ ] 6. Backend services, MCP server, ingestion
  - [x] 6.1 Control-plane services (`backend/services/experiments.py`), repository over S3 (the
    three state docs), framework-free `errors.py`, `schemas.py`, `container.py` composition
    root, `auth.py` (ownership + flagged dev identity). No queue, no local paths.
    - _Requirements: 3, 5_
  - [x] 6.2 Ingestion (`backend/services/datasets.py`): pre-signed S3 PUT + register, ported
    validation (size, format, NUL byte, duplicate headers), bounded preview; `ingest_bytes` for
    an HTTP route and tests.
    - _Requirements: 10_
  - [x] 6.3 `mcp_server/` (FastMCP 4.1.0): 13 tools wrapping the services; read-only annotations;
    `Authorizer` ownership check on every mutation; `start_training` returns an execution id at
    once; bounded previews; AppError → structured `ToolError`; no AWS passthrough.
    - _Requirements: 9.1-9.6_
  - [x] 6.4 In-process FastMCP client tests: discovery, annotations, validation, authz denial,
    bounded outputs, structured errors, and a full create→EDA→review→decide→train→compare flow
    with no AWS.
    - _Requirements: 9.7; Property 9_
  - [x] 6.5 HTTP entrypoint `mcp_server/app.py` (streamable HTTP at `/mcp`, `/health` for the
    load balancer, builds the container from `Settings`); example `infra/librechat/` config +
    README; `infra/iam/` roles from Phase 9. Auth remains the flagged dev identity until the
    mechanism is chosen (Q8); the entrypoint trusts no raw header. Tests assert health honesty
    and that the HTTP app exposes the same tool set as the in-memory one.
    - _Requirements: 9_

- [ ] 7. Business context, leakage, feature engineering
  - [x] 7.1 `BusinessContext` contract + `LeakageService.set_business_context`/`get`; plain-language
    objective + scenario replace the timing question (req 11.2); leakage investigation defaults
    on; `as_reasoning_context()` exposes only the stated intent for Bedrock, no data.
    - _Requirements: 11_
  - [x] 7.2 New `LeakageWarning`/`LeakageInvestigationReport` contracts (additive; inherited
    `leakage.py` and `reasoning.py` unchanged) with full warning fields: id, method, evidence,
    severity, confidence, reasoning, recommended investigation, correction, decision, review.
    - _Requirements: 12.1, 12.4_
  - [x] 7.3 `ml_engine/leakage/investigation.py` builds pre- and post-training phases from the
    deterministic artifacts (rules + associations + names + importance/ablation), stored as
    separate artifacts that never overwrite; association never sets `leakage_established` (12.3).
    - _Requirements: 12.2, 12.3, 12.7_
  - [x] 7.4 `decide_warning` (keep/exclude/investigate/accept-risk/compare) and a *separate*
    `apply_exclusions_to_selection`, so a leakage verdict never silently changes what trains;
    nothing auto-excluded; MCP tools wired; decisions survive a phase rebuild.
    - _Requirements: 12.5, 12.6; Property 7_
  - [ ] 7.5 **Deferred by owner priority:** new feature operators. Adding one edits the inherited
    `proposals.py` + `derive.py` (manifest-protected core) and is not the priority capability.
    The inherited 5-operator vocabulary and its validation are reused unchanged. Revisit only if
    a concrete operator is requested, with justification + tests (req 13.1).
    - _Requirements: 13_

- [x] 8. Reporting
  - [x] 8.1 `Report` contract + pure builders (`ml_engine/reporting/documents.py`) that assemble
    EDA and model-comparison reports from existing artifacts (adding no numbers), a dependency-free
    HTML renderer (`render.py`, escapes output), and `ReportingService` that writes JSON + HTML to
    S3 and a PDF **when WeasyPrint is installed** (optional `reports` extra; absent → JSON+HTML and
    `pdf_available=false`). MCP tools `generate_report`/`get_report`; brief §9 fields covered.
    Spot/cost deliberately excluded from the report (MLOps-owned, owner decision).
    - _Requirements: 14_

- [x] 9. Security and governance (cross-cutting)
  - [x] 9.1 Least-privilege IAM policies in `infra/iam/` (control plane, SageMaker job, Step
    Functions) with placeholders + README; `tests/infrastructure/test_iam_policies.py` enforces
    no wildcard actions/resources, no endpoint/model creation, PassRole scoped to SageMaker, and
    the job role cannot submit jobs or pass roles.
    - _Requirements: 15.3_
  - [x] 9.2 Auth boundary: ownership checked on every MCP mutation (Phase 6); dev-identity
    stand-in flagged loudly until the LibreChat auth mechanism is chosen.
    - _Requirements: 15.1, 15.2_
  - [x] 9.3 `backend/governance.py`: Bedrock region/profile check wired into `Settings.problems()`
    (a `us.` profile needs a `us-*` region); governance gate that caps sample-value count and
    length (configurable) or strips them, over the already-bounded reasoning context — only ever
    removes information, never a full dataset. (Applies when Bedrock reasoning is invoked.)
    - _Requirements: 15.5, 16.4_
  - [x] 9.4 Limits/audit: `backend/audit.py` append-only per-experiment trail of every MCP
    mutation (who/what/when, one object per event, never breaks the operation); upload-size and
    retention settings; audit toggle.
    - _Requirements: 15.4_

- [x] 10. Audit fixes and analysis features (2026-10-26)
  - [x] 10.1 Run integrity: previous run archived to `runs/<id>/` before retrain (no stale
    model results); Spot checkpoints scoped per run; `save_features`/`apply_leakage_exclusions`
    refused while training; datasets pinned under `input/` (also holdout and scoring inputs).
  - [x] 10.2 Orchestration: `stop()` tolerates per-job refusals and keeps observing until AWS
    confirms; models beyond the parallel limit train in waves; platform-wide cap via
    `ListTrainingJobs`; delete stops running work.
  - [x] 10.3 Wiring: governed `ReasoningService` (LLM leak notes in the pre-training phase,
    `interpret_results`); post-training phase reads permutation importance and ablation.
  - [x] 10.4 Engine (reviewed manifest change): `group` and `temporal` split strategies; the
    group split is the default when the business context names an entity id.
  - [x] 10.5 Ingestion: register limited to approved prefixes; bounded ranged reads (CSV head,
    Parquet footer); non-UTF-8 and wrong-delimiter files refused with the reason.
  - [x] 10.6 New: permutation importance (training job), ablation job, batch scoring job,
    pre-flight check, execution status + diagnoses, rerun/list runs, experiment comparison,
    pre-signed download links. 15 new MCP tools.
  - Open: EventBridge route (needs an authenticated endpoint), retention sweep
    (`data_retention_days` still unused), cross-validation / separate test fold, calibration and
    drift reports.

## Notes

- Each task ships with its tests; `make test` and `make lint` must pass before it is checked off.
- Inherited engine files change only through a reviewed manifest update.
- Correctness Properties 1–11 in `design.md` are regression gates.
