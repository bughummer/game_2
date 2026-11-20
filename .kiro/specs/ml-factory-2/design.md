# Design Document

ML Factory 2.0 — SageMaker + S3 + MCP. Grounded in `assessment.md`.

> **Superseded in places by the stock-image migration** (`.kiro/specs/stock-image-migration/`,
> decision B1–B13). Where this document and that decision conflict, the decision wins. Specifically:
> - **Q5 (ECR required) and DD-7 (one custom image):** no custom image is built or pushed. Every
>   job runs the AWS prebuilt SageMaker XGBoost image (which also ships scikit-learn), resolved at
>   submission by `backend/aws/images.py::resolve_stage_image`. ECR pull permission is still needed,
>   but for AWS's registry.
> - **C3/C4 and Req 4.3 (Spot checkpoint/resume):** dropped. `RESUMABLE_MODELS` is empty, no job
>   sets a `CheckpointConfig`, and every model is restart-on-interruption. Managed Spot (the
>   discount) is kept. The CatBoost snapshot / XGBoost resume mechanisms described below are no
>   longer used.
> - **Model set:** the built-in catalogue is sklearn + XGBoost (logistic_regression, elastic_net,
>   random_forest, xgboost, and regressor variants). CatBoost is not registered by default; it and
>   any other PyPI learner are reached via a user-approved per-experiment `requirements` install.
> - **Property 5 / Req 3.6 (reproducibility):** no longer "identical pinned versions verified against
>   the offline suite". It is the recorded image digest plus the per-experiment `pip freeze`
>   (Stage 2). The offline suite runs on whatever versions resolve locally.
> - **LLM boundary (platform-wide):** the former "the LLM never decides infrastructure" is replaced
>   by decision B12 — the LLM may propose any choice (packages, image, treatment, models) and
>   explain it, but computes no number and applies nothing without explicit user approval, which is
>   recorded in the audit trail. The two absolutes remain: the LLM computes no number, and nothing
>   it proposes takes effect without user approval.

## Overview

One store (S3), one orchestrator (SageMaker). The Protocols stay as the seams tests substitute.

## Architecture

```
LibreChat ──▶ MCP server ──▶ FastAPI backend ──▶ ExperimentOrchestrator ──▶ SageMaker jobs
(chat UI)     (typed tools,    (validate, record,   SageMakerOrchestrator      (jobs/*/main.py,
              async, authz)     submit, read)       boto3 → Step Functions)     S3 I/O)
                                   │                                              │
                                   ├──▶ ObjectStore = S3ObjectStore ──▶ S3 ◀──────┘
                                   ├──▶ tracking: status · Spot · cost · diagnostics
                                   └──▶ Bedrock (reasoning; reads artifacts, never computes)
```

### Repository layout

```
ml_factory_2/
  ml_engine/            inherited verbatim (+ io/s3.py new, io/uri.py rewritten for s3://)
  jobs/                 inherited stage CLIs; build_store() → S3ObjectStore
  backend/
    config.py           pydantic-settings; AWS/S3/SageMaker/budget/governance settings
    container.py        wires S3ObjectStore, SageMakerOrchestrator, tracking, Bedrock
    aws/                boto3 client factory (role assumption, region checks)
    orchestration/      base.py (inherited), sagemaker.py, step_functions.py
    tracking/           runs, status monitor, spot/cost, pricing table, diagnostics
    services/, api/     ported from old_repo minus queue/local
  mcp_server/           FastMCP tools (named mcp_server to avoid shadowing the `mcp` SDK)
  container/            Dockerfile for the stage image (SageMaker)
  infra/                IAM policies, Step Functions ASL, price table JSON
  tests/                mirrors source; support/ holds InMemoryObjectStore, InlineOrchestrator
```

## Components and Interfaces

### C1. S3ObjectStore (Req 2)
- `ml_engine/io/s3.py`, written new against the 8-method Protocol. Constructor takes an injected
  boto3 S3 client, `sse` (`"AES256"` or `"aws:kms"` + key id). No boto3 import in `ml_engine`.
- Semantics fixed by the shared store contract suite (run against `InMemoryObjectStore` and
  `moto`): `read_bytes` on missing key → `ObjectNotFoundError`; `list_uris(prefix)` returns full
  `s3://` URIs, paginated, sorted; `list_prefixes` uses `Delimiter="/"` and returns child prefixes
  ending in `/`; `stat` → `ObjectMetadata` with size, last-modified, ETag, version id; `download`
  /`upload` via `download_file`/`upload_file` (multipart for large datasets).
- `ml_engine/io/uri.py`: `parse_s3_uri` → `(bucket, key)`, `is_s3_uri`, `normalize_storage_root`
  (requires `s3://`), plus inherited `join_uri`, `uri_matches_prefix`, `detect_format`.
  `InvalidS3UriError` replaces `InvalidFileUriError`.
- `jobs/_common/runtime.py::build_store()` builds the boto3 client from the job environment
  (SageMaker execution role) and returns `S3ObjectStore`. The backend builds its own in
  `container.py`. These are the only two construction sites.

### C2. Stage container (Req 3.2)
- One image, `python:<same minor as the test env>-slim` base (3.14 today), installs the project
  with `jobs,models` extras at the exact pins in `pyproject.toml`.
  Entrypoint dispatches on SageMaker's `train` argument and on Processing `ContainerEntrypoint`
  to `python -m jobs.<stage>.main --experiment-id ... --artifact-base s3://...`.
- Thread counts pinned via env (`OMP_NUM_THREADS`, model `nthread`/`thread_count` passed through
  hyperparameters) for reproducibility.
- Training jobs read and write S3 directly through `S3ObjectStore` (no SageMaker input channels
  needed); `/opt/ml/checkpoints` is used only for checkpoints.

### C3. SageMakerOrchestrator (Req 3, 4)
- `backend/orchestration/sagemaker.py`. Phase A: direct boto3.
  - `start_eda` → `create_processing_job` (profiling stage).
  - `start_training` → freeze config to S3, then `create_processing_job` (preprocessing), and on
    its completion one `create_training_job` per model, then `create_processing_job`
    (evaluation). In Phase A, stage chaining is driven by the status monitor (C5) advancing the
    workflow; in Phase B, Step Functions replaces this.
  - `describe(execution_id)` returns raw + normalized status from tracking.
  - Returns `ExecutionHandle` immediately; job names are deterministic
    (`mlf-<exp8>-<stage>-<model>-<attempt>`, ≤ 63 chars) so resubmission is idempotent.
- Spot: `EnableManagedSpotTraining=True`, `StoppingCondition.MaxWaitTimeInSeconds` ≥
  `MaxRuntimeInSeconds`, `CheckpointConfig` only for XGBoost/CatBoost, `RetryStrategy` (default 1).
  On-Demand only if `experiment.compute.allow_on_demand_fallback` is true, recorded on the run.
- Pre-submission guards (pure functions, unit-tested): instance in allow-list ≤ ceiling, CPU only,
  concurrency ≤ limit, cost within per-experiment and per-month budget.
- Budget enforcement uses **reservations**: each job's upper bound (on-demand rate × instances ×
  `MaxRuntimeInSeconds`) is written once, at submission, to `spend/<job>.json` under the
  experiment and to `ml-factory/spend/<YYYY-MM>/<job>.json`. The whole workflow is reserved
  before its first job is submitted. Unknown price or an unverified table refuses submission.
  Reservations are conservative (never-submitted jobs keep theirs) until tracking reconciles
  them with measured billable time (task 4.3). The read-check-write is serialized by an
  in-process lock; multiple control-plane replicas would need a conditional-write ledger.
- Processing jobs run On-Demand (SageMaker Managed Spot applies to Training jobs only); the
  Spot-only rule therefore applies to training.
- Phase B: `backend/orchestration/step_functions.py` — state machine `Prepare → Map(Train per
  model, Catch per item) → Evaluate`, using the native `.sync` SageMaker integrations. Same
  Protocol; switching is one line in `container.py`. Approval gates use task tokens.

### C4. Checkpointing (Req 4.3)
Verified against the inherited plugins: both pass user overrides through `resolve_params`, but
`fit` calls `estimator.fit(x, y)` with no resume argument.
- **CatBoost** — fully outside the engine: the training stage passes overrides
  `save_snapshot=True`, `snapshot_file=/opt/ml/checkpoints/catboost.snapshot`,
  `train_dir=/opt/ml/checkpoints`, `allow_writing_files=True`. CatBoost resumes from the snapshot
  automatically on restart.
- **XGBoost** — checkpoint writing is an override (`callbacks=[TrainingCheckPoint(...)]`), but
  resuming needs `fit(..., xgb_model=<checkpoint>)`. Plan: a resume wrapper in `jobs/training`
  around the built estimator (forwarding `xgb_model` through `LabelEncodedClassifier`). If that
  cannot be done without editing `ml_engine/models`, the fallback is a small additive
  `resume_from` parameter on the two XGBoost plugins — this is an engine change and needs owner
  approval (open question Q3).
- Overrides apply only on SageMaker Spot runs; with no checkpoint present, results must equal the
  plain plugin path (asserted by test).
- A resumed run is **not** guaranteed to match an uninterrupted run (RNG state for subsampling
  restarts). `RunRecord.resumed_from_checkpoint` is recorded and the reproducibility property
  applies only to non-resumed runs.
- Resumability table (`backend/tracking/resumability.py`): xgboost, xgboost_regressor, catboost,
  catboost_regressor → resumable; logistic_regression, elastic_net, random_forest,
  random_forest_regressor → restart-on-interruption.

### C5. Status monitor and failure diagnostics (Req 6, 8)
Scope reduced by owner: **no Spot/cost recording** (requirement 7 is MLOps's domain). This
component answers "where is my job?" and "why did it fail?", nothing about money.
- Contracts (new, in `ml_engine/contracts/tracking.py`): `NormalizedStatus` (enum incl.
  `UNKNOWN`), `JobObservation` (job name, raw status, secondary status, failure reason,
  timestamps, instance config — all AWS-reported facts, no prices), `FailureCategory`,
  `FailureDiagnosis`.
- Layout additions: `executions/<id>/observations/<job>.json`, `executions/<id>/diagnosis/
  <job>.json`. One writer each: the monitor writes observations, the diagnoser writes diagnoses.
- Status monitor: an EventBridge rule on SageMaker job state changes invokes a backend endpoint
  (or a small Lambda calling it); a polling fallback (`describe_*` with backoff) covers missed
  events. Both persist an observation and call `SageMakerOrchestrator.advance()`. Normalization
  is a pure function of raw fields; insufficient evidence → `UNKNOWN`, never a guess.
- Diagnostics: ordered field/substring rules over `FailureReason` + the stage's `failure.json` +
  bounded CloudWatch logs (`filter_log_events`, ≤ 200 lines / 32 KB) → a category; redaction
  (emails, AWS keys, account ids, ARNs, IPs, long digit runs) before any text could reach an LLM.

### C6. Backend (ported)
- Port `old_repo/backend` services/api/schemas/repositories module by module, deleting queue,
  `queue_dir`, `file://` and local-mode settings. `Settings.validate()` requires `artifact_root`
  to be `s3://`, a SageMaker execution role ARN, an image URI and a region.
- No heavy compute in request handlers (bounded exception: reading an upload's header row).

### C7. MCP server (Req 9, 10)
- `mcp_server/` with FastMCP. Tool groups: experiments (create, get, list), data
  (request_upload_url, register_dataset, preview — bounded rows/columns), analysis (start_eda,
  get_eda_summary, get_leakage_findings), features (propose, validate, decide), training
  (start_training, get_status, stop), results (get_comparison, get_run, get_costs, get_report).
- Each tool calls one backend service function, returns contracts. Annotations mark
  `readOnlyHint`; mutations check authorization via an injected `Authorizer`.
- Transport: streamable HTTP for LibreChat; tests use FastMCP's in-memory client.

### C8. Business context, leakage, features, reporting (Req 11–14)
- `BusinessContext` contract stored at `config/business_context.json`; replaces the timing
  question; read by leakage, feature proposals and reasoning prompts.
- Leakage findings: new `pre_training` and `post_training` artifacts
  (`leakage/pre_training.json`, `leakage/post_training.json`); inherited `eda/leakage.json`
  stays as the deterministic-rules output. Extended warning fields added as a new contract
  version; inherited rules unchanged.
- New feature operators and reporting libraries follow the inherited closed-vocabulary pattern,
  each with validation + tests.

## Data models

All new records are versioned Pydantic contracts in `ml_engine/contracts`, reused verbatim by API
and MCP. Paths only in `ml_engine/io/layout.py`.

## Error handling

- Stage failures → `failure.json` + non-zero exit (inherited) → SageMaker `Failed` → diagnosis.
- Guard failures (budget, instance, concurrency) reject before submission with a typed error.
- AWS client errors are mapped to typed backend errors; throttling retried with botocore
  standard retry mode.
- Unknown is a first-class value for status, cost and savings.

## Testing strategy

- Default suite: no network, no AWS. `InMemoryObjectStore` and `InlineOrchestrator` (runs stage
  `main` functions in-process against the in-memory store) under `tests/support/`; `moto` for
  S3/CloudWatch Logs; botocore `Stubber` for SageMaker and Step Functions request shapes.
- Store contract suite parametrized over both stores.
- Opt-in real AWS: `ML_FACTORY_AWS_TESTS=1` + configured profile; includes the reproducibility
  test (two SageMaker runs, same frozen config, tolerance below).
- Inherited-file manifest test (Req 1.5) and architecture-boundary test.
- ruff for lint/format; property-based tests seeded.

## Design decisions

- **DD-1 — boto3 first, Step Functions second, never SageMaker Pipelines.** Unchanged from the
  reference design.
- **DD-2 — write `S3ObjectStore` fresh.** Git recovery is impossible (no history). The Protocol is
  small and the shared contract suite pins its semantics.
- **DD-3 — per-run records + one aggregation writer, no database.**
- **DD-4 — cost/Spot numbers are deterministic tracking artifacts; static price table for v1.**
- **DD-5 — MCP wraps services; no business logic in tools.**
- **DD-6 — no local execution; offline tests via mocks.** One production backend removes the
  store/orchestrator selection logic entirely. Tests substitute at the Protocols.
- **DD-7 — one stage image for Processing and Training.** Fewer images to build, scan and pin;
  reproducibility depends on a single digest recorded on every run.
- **DD-8 — event-driven status with polling fallback.** EventBridge gives timely transitions;
  polling guarantees eventual consistency if an event is lost.

## Correctness properties

1. **Core purity** — `ml_engine` imports no `fastapi`, `backend`, `jobs`, `mcp_server`, SageMaker
   SDK, and no `boto3` at import time. *Req 1.3, 1.4*
2. **Fit on training fold only** — inherited preprocessing/leakage tests. *Req 13.3*
3. **Numbers only from engine/tracking** — reasoning + tracking tests. *Req 7.6, 16.1, 16.3*
4. **S3 round-trip identity** — a contract written and read through `S3ObjectStore` is
   byte-identical, and both stores pass the same contract suite. *Req 2.5, 2.6*
5. **Reproducibility within tolerance** — two SageMaker runs of one frozen config agree on the
   primary metric within 1e-6 abs or 0.1% rel (opt-in real-AWS test; in-process equivalent in the
   default suite). *Req 3.6*
6. **Single writer per document** — concurrent-run tests. *Req 5.5*
7. **Nothing auto-excluded but the target** — feature-decision tests. *Req 12.5, 12.6*
8. **LLM output schema-validated and column-checked** — reasoning tests. *Req 16.2*
9. **Non-blocking orchestration** — `start_*` and MCP tools return an id without waiting. *Req 3.3, 9.4*
10. **No fabricated AWS facts** — status/cost/savings are Unknown when evidence is missing. *Req 6.4, 7.2, 7.3, 7.5*
11. **No unapproved spend** — no job is submitted that violates instance, concurrency or budget
    guards, and On-Demand never occurs without the opt-in flag. *Req 4.4, 4.5, 4.7*

## Owner decisions (2026-10-14, cont.)

- **Cost/budget is owned by the MLOps team** (AWS Budgets, Cost Explorer, the Canvas cost
  tooling). ML Factory's own budget guard is therefore **off by default** (`enforce_budget=False`):
  it does not price or cap jobs, writes no spend ledger, and needs no price table. The safety
  guards that are *not* about spend — instance allow-list, CPU-only, Spot-only training,
  concurrency cap — always run. Setting `enforce_budget=True` turns on pricing, the per-experiment
  and monthly ceilings, and the spend ledger as an optional second layer (then an unverified or
  missing price table fails closed). Supersedes Q9: price-table verification is only needed if
  MLOps opts ML Factory into enforcement.

## Owner decisions (2026-10-14)

- Approved: fresh `S3ObjectStore` (DD-2); resumed runs excluded from reproducibility; manifest
  test, spend-guard property, event-driven status, single stage image; no React UI in v1.
- **Q3 approved:** if the XGBoost resume wrapper cannot work outside the engine, a small additive
  `resume_from` parameter on the two XGBoost plugins is allowed.
- **Q5:** ECR is required — the engine needs Python 3.12 and CatBoost, which AWS's prebuilt
  SageMaker framework images do not provide. Account/region/bucket/ECR repo to be supplied
  before the first real-AWS test.
- **Q7:** control plane on ECS Fargate (private, behind an internal ALB). Network path to
  LibreChat to be confirmed with infra.
- The AWS API MCP Server (awslabs) is rejected as the user-facing MCP (conflicts with Req 9.6 and
  the spend/ownership guards); allowed as a read-only developer tool.

## Open questions for the owner

- **Q1** Budget ceilings (placeholders $25 / $500). *Pending from owner.*
- **Q2** Spot `MaxWaitTimeInSeconds` (default 3600) and instance ceiling (default
  `ml.m5.2xlarge`). *Pending from owner.*
- **Q4** Bedrock model id + region, sample-value cap (default 5 × 64 chars). *Pending from owner.*
- **Q9** Verify `infra/pricing/sagemaker_prices.json` against the AWS SageMaker pricing page
  for the target region and set `verified: true`. Until then every submission is refused by
  design (`PRICE_TABLE_UNVERIFIED`).
- **Q8** Backend shape: (a) separate FastAPI service, or (b) one Fargate app — service package +
  MCP tools + a few HTTP routes, with a Lambda writing SageMaker status events to S3
  (recommended). Not needed until Phase 6.
