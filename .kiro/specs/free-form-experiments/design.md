# Design: Free-form experiments

## Overview

Add a **Free_Form_Tier** as a new side workflow: the platform runs a user-approved Python script as
one SageMaker Processing job, with the experiment's `requirements` installed first, and saves the
script, the `pip freeze`, the logs, the results and (optionally) a model as its own artifacts. The
managed workflow is untouched; a free-form run never changes the experiment's managed status.

The central design fact, from reading the code: the platform already runs **side jobs**
(validation, holdout, ablation, scoring) as single Processing jobs that are members of
`SIDE_WORKFLOWS` and never mutate experiment state. A free-form run is exactly this shape — a new
`WorkflowKind.FREE_FORM` side job, a `FreeFormService` mirroring `ValidationService`, a
`jobs/free_form/main.py` entry point, and a `free_form/<run_id>/` artifact tree in the layout. The
only genuinely new pieces are: the **approved-script record** (content-hash-bound), the
**Platform_SDK** the script imports (a thin, versioned façade over the engine's existing splitters,
metrics and a results-writer — no parallel implementation), the **honest-labelling** of results
(`free_form` / self-reported vs engine), the **leakage screen** on the run's inputs (reusing
`analyze_leakage` unchanged), and the **data-context Q&A** the AI answers before drafting.

Because the platform does not control what the script does, it does not claim the managed
guarantees; it labels results instead. Trust is restored only two ways: numbers produced *through
the Platform_SDK*, or *independent validation* of the run's Model_Version through the existing
validation flow. Both are engine-computed.

When no free-form run exists, every managed artifact is byte-for-byte today's.

## Architecture

```
AI drafting (control plane, LLM boundary B12):
  user describes experiment
    └─▶ AI asks data-context questions (Req 14) ──▶ answers stored on the run
         └─▶ AI returns a Script_Draft + proposed requirements (sees profile summary, not rows)
              └─▶ user edits / instructs revisions (each retained, diffed)
                   └─▶ user APPROVES exact code  ──▶ Approved_Script (content hash)

leakage screen (before approval, a Processing job, Req 13):
  dataset + target + declared features ──▶ analyze_leakage ──▶ LeakageReport attached to the run
                                                              (advisory; never auto-edits)

free-form run (a SIDE_WORKFLOWS Processing job):
  FreeFormService.start_run(approved_script_hash, dataset, requirements, instance, spot)
    └─▶ guards (budget, concurrency, runtime cap, instance allow-list incl. GPU for free-form)
         └─▶ write free_form/<run_id>/request.json ; pin script + dataset under the run prefix
              └─▶ orchestrator.start_free_form ──▶ jobs/free_form/main.py
                   • install requirements on the stock image (B10)
                   • run the Approved_Script with INPUT_DIR / OUTPUT_DIR / SDK available
                   • capture logs, pip freeze, results
                   • if a Predict_Entry_Point is present, register a Model_Version-shaped metadata
                   └─▶ write free_form/<run_id>/result.json (labelled free_form; metrics labelled)

independent validation (reuse the existing flow, Req 7):
  the run's Model_Version ──▶ ValidationService ──▶ Engine_Metric results
```

## Components and Interfaces

### C1. Free-form contracts (`ml_engine/contracts/free_form.py`, NEW)

- `MetricLabel` enum: `self_reported`, `engine`.
- `ScriptRevision` (versioned): `revision`, `content` (or a stored URI + `content_hash`),
  `authored_by` (`ai` | `user`), `instruction` (the revise prompt, when AI-authored), `created_at`.
- `ApprovedScript` (versioned): the `content_hash`, the approving `user`, `approved_at`, the entry
  module/function names, and the declared `predict_entry_point` (optional). Execution binds to this
  hash (Req 2.4).
- `DataContextAnswers` (versioned): the Req-14 question set with per-question
  `answer | "dont_know" | "skipped"`, reusable as defaults for later runs on the same dataset.
- `FreeFormRequest` (versioned, `free_form/<id>/request.json`): `run_id`, `experiment_id`,
  `approved_script` (`ApprovedScript`), `dataset_uri`, `file_format`, `requirements`
  (`RequirementsSpec`), `compute` (`ComputeSpec`), `declared_feature_columns`, `target_column`
  (optional — unsupervised runs have none), `label_columns_for_analysis` (optional, Req 13.7),
  `declared_external_urls`, `data_context` (`DataContextAnswers`), `requested_at`, `requested_by`,
  `execution_id`.
- `FreeFormMetric` (versioned): `name`, `value`, `label: MetricLabel` — every number carries its
  label (Req 3.1).
- `FreeFormResult` (versioned, `free_form/<id>/result.json`, written only by the job): `run_id`,
  `tier="free_form"`, `status`, `metrics: list[FreeFormMetric]`, `produced_model` (bool +
  version id), `pip_freeze` ref, `image_digest`, `sdk_version`, `logs` ref, `naive_baseline`
  (time-series, Req 5.5), `declared_external_urls`, `warnings`.
- All frozen additively; no managed contract changes except the optional additions in C7.

### C2. The Platform_SDK (`ml_engine/sdk/__init__.py`, NEW façade over existing engine code)

- A thin, versioned module a script imports: `from ml_engine.sdk import splits, metrics, io`. It
  **re-exports** the engine's existing pure code (no parallel implementation, Req 4.3):
  - `splits`: `random`, `stratified`, `group`, `temporal`, and an **expanding-window
    (rolling-origin)** split for time series (Req 5.1), all over `ml_engine/splitting`.
  - `metrics`: the classification/regression metric functions and forecasting metrics (MAE, RMSE,
    MAPE/sMAPE, seasonal-naive baseline, Req 5.2) over `ml_engine/evaluation`.
  - `io.write_results(metrics, *, sdk=True)`: writes the platform's result format to `OUTPUT_DIR`,
    tagging SDK-produced metrics so the job labels them `engine` (Req 4.2).
- `SDK_VERSION` constant, recorded on every run (Req 4.4). The SDK imports only base-image
  libraries or the run's requirements (Req 4.4).
- The temporal and rolling-origin splitters **refuse** a configuration that would place a later
  observation in training than in evaluation, with a structured error (Req 5.4); a property test
  guarantees train-before-eval (Req 12.4).
- The forecasting metrics always compute a **seasonal-naive baseline** so a model that fails to beat
  it is visible (Req 5.5).

### C3. The free-form job (`jobs/free_form/main.py`, NEW)

- Entry point via `jobs._common.runtime.base_parser`/`run_entrypoint`, argument `--run-id`,
  `failure_uri = layout.free_form_failure(run_id)`.
- `run_free_form(store, layout, *, experiment_id, run_id)`: reads `free_form_request(run_id)`,
  loads the pinned Approved_Script (verifying its `content_hash`, Req 2.4), makes `INPUT_DIR`
  (the pinned dataset) and `OUTPUT_DIR` (under the run prefix) available, and **executes the
  script** with the Platform_SDK importable. A failing install or script fails the run with a
  structured `JobError` and captured logs — no fallback (Req 1.6).
- After the script: capture `pip freeze` (B11), the image digest, the SDK version; read the
  script's written results; label each metric `engine` only if the script wrote it through the SDK
  results-writer, else `self_reported` (Req 3.1, 4.2). If the script produced a model with a
  declared `Predict_Entry_Point`, write a `ModelMetadata`-shaped `metadata.json` + the model under
  the run's `models/<name>/` so it is a Model_Version (Req 6.3); reject a predict entry point that
  does not load or returns the wrong shape (Req 7.3).
- Writes `free_form_result(run_id)`. Imports only `ml_engine` + the store, never the control plane
  (the architecture-boundary rule).

### C4. The leakage-screen job (reuse; `jobs/free_form/leakage.py` or fold into the run)

- Before approval, a Processing job runs `analyze_leakage` on the run's dataset, target and declared
  feature columns, reusing the managed rules and `LeakageConfig` unchanged (Req 13.1). It needs an
  `EdaReport`; the screen profiles the free-form dataset the same way `jobs/profiling/main.py` does
  (reusing `profile_dataset`), then calls `analyze_leakage`. With no target it runs only the
  target-free checks (duplicate rows/entities, identifier-like columns) and records which were
  skipped and why (Req 13.6). When labels are provided for analysis only, it screens the features
  against those labels (Req 13.7). The `LeakageReport` is attached to the run and shown before
  approval; it is advisory and never auto-edits (Req 13.2, 13.3). Approving with open high-severity
  findings records the acknowledgement (Req 13.4).

### C5. FreeFormService (`backend/services/free_form.py`, NEW; mirrors ValidationService)

- `request_data_context(experiment_id)`: returns the Req-14 question set (seeded from the profile
  summary and column names, plus saved defaults for the dataset). Pure control-plane; no raw rows.
- `draft_script(experiment_id, *, description, data_context, previous_revision=None,
  instruction=None)`: records the answers on the run, asks the **LLM** for a `ScriptRevision`
  (the LLM sees the profile summary + answers, never rows), stores it with a diff against the
  previous revision, and returns it. The LLM may not approve or submit (Req 2.5, B12).
- `approve_script(experiment_id, run_id, *, revision, user)`: binds an `ApprovedScript` to the
  revision's content hash; any later edit needs a new approval (Req 2.4).
- `start_run(experiment_id, *, approved_script, dataset_uri, requirements, instance_type, use_spot,
  ...)`: guards (budget/concurrency/runtime-cap + the free-form instance check, C8), pins the
  script and dataset under the run prefix, writes `request.json`, calls
  `orchestrator.start_free_form(record, run_id)`, writes back the `execution_id`, returns status.
- `run_status(...)`, `list_runs(...)`: read `request.json` + `result.json`/`failure.json` +
  execution, like `ValidationService.validation`.
- `validate_run_model(experiment_id, run_id, dataset_uri, ...)`: thin pass-through to the existing
  `ValidationService` for the run's Model_Version (Req 7.1); engine metrics (Req 7.2).
- Ownership + audit at the API layer; every draft revision, approval, submission, stop and outcome
  is audited (Req 2.6, 10.3).

### C6. Orchestration (`backend/orchestration/`, inherited)

- `backend/orchestration/base.py`: add `start_free_form(record, run_id)` to the
  `ExperimentOrchestrator` Protocol and the inline test orchestrator.
- `backend/orchestration/sagemaker.py`: add `"free_form": "jobs.free_form.main"` to
  `STAGE_MODULES` and a `start_free_form` wrapping `_start_side_job(record,
  WorkflowKind.FREE_FORM, "free_form", ["--run-id", run_id], image_uri=...)`. Because
  `WorkflowKind.FREE_FORM` is in `SIDE_WORKFLOWS`, `advance()`/`stop()` route it as a side job and
  never touch managed status. The run honours `use_spot` (Req 1.5) via its `ComputeSpec`.

### C7. Contracts, layout, model-version provenance (inherited)

- `ml_engine/contracts/execution.py`: add `WorkflowKind.FREE_FORM` and include it in
  `SIDE_WORKFLOWS`.
- `ml_engine/io/layout.py`: add the `free_form/<run_id>/` tree — `free_form_prefix`,
  `free_form_request/result/failure(run_id)`, `free_form_script(run_id)`,
  `free_form_input_dir(run_id)`, `free_form_leakage(run_id)`, and a `free_form_model_dir(run_id,
  name)` resolving the Model_Version-shaped artifacts. These survive a retrain (like `validations/`),
  so they are **not** added to `RUN_OUTPUT_PREFIXES`.
- `ml_engine/contracts/model_version.py`: extend `ModelVersion.source` Literal with `"free_form"`
  (additive). `backend/services/model_versions.py`: enumerate free-form run models alongside the
  managed `models/<name>/` tree so a free-form model is listable/validatable (Req 6.3, 7.1).

### C8. Guards for free-form instances (`backend/orchestration/guards.py`, inherited)

- Req 1.4 allows **any** supported instance (CPU or GPU) for a free-form run, but the managed
  `check_instance` blocks GPU families. Add a `check_free_form_instance(settings, instance_type)`
  that enforces the allow-list and the budget but **permits** the GPU families when the instance is
  in a new `settings.allowed_free_form_instance_types` (off-list by default, opt-in). Managed
  `check_instance` is unchanged, so managed jobs stay CPU-only. The budget/concurrency/runtime
  guards apply unchanged (Req 10.1).

### C9. Request + surfacing (`backend/schemas.py`, `mcp_server`, `backend/api`)

- API/MCP shapes: `DataContextQuestion`/`DataContextAnswers`, `DraftScriptRequest`,
  `ApproveScriptRequest`, `StartFreeFormRequest`. Routes under
  `/experiments/{id}/free-form/...`: request questions, draft, approve, start, status, list,
  validate. Each mutation: `principal` -> `require_owner` -> service -> `audit_mutation`.
- A comparison endpoint lists free-form and managed runs together, each with its `tier` and metric
  labels, and offers a like-for-like comparison on engine metrics only, stating when two entries are
  not comparable (Req 8).

## Data Models

New `ml_engine/contracts/free_form.py` (above). `ModelVersion.source` gains `"free_form"`. New layout
paths under `free_form/<run_id>/`. No existing path or default changes, so a no-free-form experiment
is byte-identical. The Platform_SDK is code, not a contract, but carries a `SDK_VERSION` recorded on
each run.

## Correctness Properties

### Property 1: only an approved, unmodified script runs
A run executes only the exact bytes whose `content_hash` an explicit user approval bound; a modified
or unapproved script is refused, and the AI path can neither approve nor submit.
**Validates: Requirements 2.4, 2.5, 12.2**

### Property 2: honest labelling
Every free-form result is labelled `free_form`; each metric is `self_reported` unless produced
through the Platform_SDK or by independent validation, and self-reported numbers are never merged
into an engine-metric ranking column by default.
**Validates: Requirements 3.1, 3.2, 3.3, 12.3**

### Property 3: temporal safety
Every SDK temporal / rolling-origin split places all training rows strictly before all evaluation
rows (with the optional gap); a violating configuration is refused.
**Validates: Requirements 5.1, 5.4, 12.4**

### Property 4: SDK equals managed
An SDK-produced metric equals the managed pipeline's metric on the same inputs (same code path).
**Validates: Requirements 4.2, 4.3, 12.5**

### Property 5: independent validation
A run exposing a Predict_Entry_Point registers a Model_Version that validates through the standard
flow with engine metrics; a run with none keeps self-reported results and says validation is
unavailable.
**Validates: Requirements 6.3, 7.1, 7.2, 7.4, 12.5**

### Property 6: isolation and no-fallback
One run never reads or overwrites another's outputs; a failed install or failing script fails the
run with a structured error and no fallback.
**Validates: Requirements 1.6, 1.7, 12.6**

### Property 7: leakage screen is advisory
The free-form dataset is leakage-screened before approval; findings attach to the run and are shown,
and a finding never blocks or edits the run on its own.
**Validates: Requirements 13.1, 13.2, 13.3, 12.7**

### Property 8: data-context gate
A Script_Draft cannot be produced until the data-context questions are answered or explicitly
skipped; the answers/skips are stored with the run and passed to drafting.
**Validates: Requirements 14.1, 14.3, 14.7, 12.8**

### Property 9: backward compatibility
An experiment with no free-form run produces byte-identical managed artifacts.
**Validates: Requirements 11.2**

## Error Handling

- A run whose script's content hash does not match the approval → `ValidationError`
  (`SCRIPT_NOT_APPROVED`) before submission.
- A drafting call before the data-context questions are answered/skipped → `ConflictError`
  (`DATA_CONTEXT_REQUIRED`).
- An SDK temporal config that leaks future into past → `ValidationError` (`TEMPORAL_LEAK`) from the
  SDK, inside the job, failing the run with captured logs.
- A declared Predict_Entry_Point that does not load or returns the wrong shape → the registration is
  rejected and the run keeps self-reported results (Req 7.3, 7.4).
- Over budget / concurrency / runtime cap, or a disallowed instance → the existing `GuardViolation`
  with the estimate and the ceiling (Req 10.1).
- A failed install or failing script → structured `JobError` with captured logs; no fallback
  (Req 1.6).

## Testing Strategy

Offline, `InMemoryObjectStore` + `InlineOrchestrator` + mocked AWS; the free-form job runs a tiny
script fixture in-process.

- An unapproved/modified script is never executed; approval binds to the content hash; the AI path
  cannot approve or submit (Property 1).
- A result is labelled `free_form` + self-reported unless produced through the SDK or validation
  (Property 2).
- A Hypothesis property test that every SDK temporal split is train-before-eval, and a refusal test
  for a leaking config (Property 3).
- An SDK metric equals the managed metric on the same inputs (Property 4).
- A Predict_Entry_Point model validates through the standard flow; a run without one says validation
  is unavailable (Property 5).
- Run isolation (no cross-run overwrite) and a failed install failing the run without fallback
  (Property 6).
- The free-form dataset is leakage-screened before approval; findings attach and never auto-edit
  (Property 7).
- A Script_Draft cannot be produced until the data-context questions are answered/skipped, and the
  answers are stored and passed to drafting (Property 8).
- A no-free-form experiment is byte-identical to today (Property 9, golden).

## What changes in inherited files (manifest)

| File | Change | Justification |
|---|---|---|
| `ml_engine/contracts/execution.py` | Add `WorkflowKind.FREE_FORM`; include in `SIDE_WORKFLOWS` | Route a free-form run as a side job that never touches managed status |
| `ml_engine/io/layout.py` | Add the `free_form/<run_id>/` tree | New additive side-job paths, defined only here |
| `ml_engine/contracts/model_version.py` | Extend `ModelVersion.source` with `"free_form"` | Provenance of a free-form Model_Version (Req 6.3) |
| `backend/services/model_versions.py` | Enumerate free-form run models too | Make a free-form model listable/validatable |
| `backend/orchestration/base.py`, `backend/orchestration/sagemaker.py` | Add `start_free_form` + the `STAGE_MODULES` entry | Submit the free-form Processing job |
| `backend/orchestration/guards.py` | Add `check_free_form_instance` (GPU opt-in) | Req 1.4 allows GPU for free-form; managed stays CPU-only |
| `backend/schemas.py` | Free-form request/response shapes | Surface the free-form flow |
| `backend/container.py`, `backend/api/app.py` | Construct + register `FreeFormService` and its routes | Wire the new service |
| `mcp_server/server.py` | Free-form tools (draft/approve/start/status), echoing labels | Chat-side surface under B12 |

New modules (`contracts/free_form.py`, `sdk/*`, `jobs/free_form/*`, `services/free_form.py`,
`api/routes/free_form.py`) are registered as NEW. Each inherited edit is additive and recorded in
`tests/inherited_manifest.json` with a written justification.

## Phasing

1. **Run + save + label** (C1, C3 minus SDK labelling nuance, C5 start/status, C6, C7, C8): run an
   approved script, save everything, label results `free_form`/self-reported.
2. **Platform_SDK** (C2) + engine-metric labelling + time-series splits/metrics.
3. **Independent validation** (C5 pass-through, model-version registration) + comparison (C9).
4. **Leakage screen** (C4) and **data-context Q&A** (C5 draft gate) — the trust bridges.

## Open questions (resolved in design)

- **Time series scope (Q1):** both — temporal + rolling-origin splits and forecasting metrics in the
  SDK; deeper forecasting tooling later.
- **Sweeps (Q2):** out; one approved script is one run.
- **Runtime/cost cap (Q3):** existing budget guards plus a configurable per-run runtime cap.
- **SDK by default (Q4):** yes; AI drafts use the SDK for splitting and metrics unless the user asks
  otherwise.
- **Internet access (Q5):** allowed via the existing proxy; declared external URLs are recorded on
  the run.
