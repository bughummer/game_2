# Implementation Plan: Free-form experiments

## Overview

Python, following `design.md`. The core insight: a free-form run is a new side workflow — the same
shape as the existing validation/holdout/scoring side jobs — so most of the machinery is reused. A
new `WorkflowKind.FREE_FORM` in `SIDE_WORKFLOWS`, a `free_form/<run_id>/` layout tree, a
`jobs/free_form/main.py` entry point, a `FreeFormService` mirroring `ValidationService`, and a
Platform_SDK that is a thin versioned façade over the engine's existing splitters/metrics (no
parallel implementation). Build phased: run+save+label first, then the SDK + time series, then
independent validation + comparison, then the leakage screen and the data-context Q&A. Every
inherited-file change is additive and followed by a manifest update. A no-free-form experiment stays
byte-for-byte today's. Commands: `.venv\Scripts\python -m pytest -q -p no:cacheprovider`,
`.venv\Scripts\python -m ruff check .`, `.venv\Scripts\python -m ruff format --check .`.

## Task Dependency Graph

```
1 free-form contracts (NEW) + WorkflowKind.FREE_FORM + layout tree + model_version source (inherited)
└─▶ 2 orchestration: start_free_form + STAGE_MODULES + free-form instance guard (inherited)
     └─▶ 3 the free-form job: run approved script, capture, label, register model (NEW)
          └─▶ 4 FreeFormService: approve/start/status + request field; routes (NEW + inherited)
               └─▶ 5 checkpoint
                    └─▶ 6 Platform_SDK: splitters + metrics façade + time series + engine labelling
                         └─▶ 7 independent validation pass-through + model-version enumeration + compare
                              └─▶ 8 leakage screen on inputs (reuse analyze_leakage)
                                   └─▶ 9 data-context Q&A gate + AI drafting (service + surfacing)
                                        └─▶ 10 end-to-end + backward-compat + golden tests
                                             └─▶ 11 final checkpoint
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
    { "wave": 8, "tasks": ["8"], "depends_on": ["7"] },
    { "wave": 9, "tasks": ["9"], "depends_on": ["8"] },
    { "wave": 10, "tasks": ["10"], "depends_on": ["9"] },
    { "wave": 11, "tasks": ["11"], "depends_on": ["10"] }
  ]
}
```

## Tasks

- [x] 1. Contracts, workflow kind, layout tree, model-version provenance (design C1, C7)
  - `ml_engine/contracts/free_form.py` (NEW): `MetricLabel`, `ScriptRevision`, `ApprovedScript`
    (content-hash bound), `DataContextAnswers`, `FreeFormRequest`, `FreeFormMetric`,
    `FreeFormResult`. All versioned
  - `ml_engine/contracts/execution.py`: add `WorkflowKind.FREE_FORM` and include it in
    `SIDE_WORKFLOWS` (routes it as a side job that never changes managed status)
  - `ml_engine/io/layout.py`: add the `free_form/<run_id>/` tree (`free_form_prefix`,
    `free_form_request/result/failure/script/input_dir/leakage`, `free_form_model_dir`); survives a
    retrain (not in `RUN_OUTPUT_PREFIXES`)
  - `ml_engine/contracts/model_version.py`: extend `ModelVersion.source` Literal with `"free_form"`
  - Register `contracts/free_form.py` as NEW; manifest updates + justifications for execution.py,
    layout.py, model_version.py; regenerate
  - Tests: every new contract round-trips; FREE_FORM is a side workflow; the layout paths are
    distinct per run id; a config/experiment with no free-form run is unaffected
  - _Requirements: 1.1, 11.1, 11.2, 11.3_

- [x] 2. Orchestration: start_free_form + stage module + free-form instance guard (design C6, C8)
  - `backend/orchestration/base.py`: add `start_free_form(record, run_id)` to the
    `ExperimentOrchestrator` Protocol; add it to the inline test orchestrator in `tests/support/`
  - `backend/orchestration/sagemaker.py`: add `"free_form": "jobs.free_form.main"` to
    `STAGE_MODULES` and `start_free_form` wrapping `_start_side_job(record,
    WorkflowKind.FREE_FORM, "free_form", ["--run-id", run_id], image_uri=...)`; honour `use_spot`
  - `backend/orchestration/guards.py`: add `check_free_form_instance(settings, instance_type)` that
    allow-lists (incl. GPU when in a new `settings.allowed_free_form_instance_types`, opt-in) and
    keeps the budget guard; managed `check_instance` unchanged (CPU-only)
  - `backend/config.py`: add `allowed_free_form_instance_types` (empty by default) and a per-run
    runtime cap setting
  - Manifest updates + justifications; regenerate
  - Tests: FREE_FORM routes as a side job (no managed status change); a GPU instance is refused
    unless opted in; the budget/runtime guards still apply
  - _Requirements: 1.2, 1.4, 1.5, 10.1_

- [x] 3. The free-form job: run the approved script, capture, label, register a model (design C3)
  - `jobs/free_form/main.py` (NEW): `--run-id` entry point; `run_free_form` reads the request,
    verifies the Approved_Script content hash (Req 2.4), exposes INPUT_DIR/OUTPUT_DIR and runs the
    script; captures `pip freeze` (B11), image digest, SDK version, logs; reads the script's
    results and labels each metric `engine` iff written through the SDK results-writer else
    `self_reported`; if a Predict_Entry_Point is declared, writes a ModelMetadata-shaped
    metadata.json + model under the run's models/<name>/ (Req 6.3), rejecting an entry point that
    does not load or returns the wrong shape (Req 7.3). A failed install/script fails the run with a
    structured JobError + logs, no fallback (Req 1.6). Imports only ml_engine + the store
  - Register `jobs/free_form/main.py` + `__init__.py` as NEW; regenerate
  - Tests (job in-process on a tiny script fixture): the run writes a labelled `free_form` result;
    a hash mismatch is refused; a failing script fails with captured logs and no fallback; two runs
    do not overwrite each other (Property 1/2/6)
  - _Requirements: 1.3, 1.6, 1.7, 3.1, 6.1, 6.3, 6.4, 7.3_

- [x] 4. FreeFormService + routes: approve, start, status (design C5, C9)
  - `backend/services/free_form.py` (NEW): `approve_script` (binds ApprovedScript to a revision's
    content hash), `start_run` (guards, pin script + dataset under the run prefix, write request,
    orchestrator.start_free_form, write back execution id), `run_status`, `list_runs` (mirror
    ValidationService)
  - `backend/schemas.py`: `ApproveScriptRequest`, `StartFreeFormRequest` (+ response shapes)
  - `backend/container.py`: construct + register the service on `AppContainer`
  - `backend/api/routes/free_form.py` (NEW) + register: POST approve / POST start / GET status / GET
    list; each mutation principal -> require_owner -> service -> audit_mutation (Req 10.2, 10.3)
  - Register the new modules as NEW; manifest updates for container/app/schemas if hash-tracked;
    regenerate
  - Tests (in-process client): start a run from an approved script end to end; a run with an
    unapproved hash is refused; ownership enforced; the submission is audited
  - _Requirements: 2.4, 2.5, 2.6, 10.2, 10.3, 10.4_

- [x] 5. Checkpoint - Ensure all tests pass
  - `.venv\Scripts\python -m pytest -q -p no:cacheprovider`, `ruff check .`, `ruff format --check .`;
    confirm `tests/test_inherited_manifest.py` + `tests/test_architecture_boundaries.py` pass; ask
    the user if questions arise
  - Result: contracts + workflow kind + layout + orchestration + the job + the service in place;
    1550 passed, 1 skipped; ruff clean; manifest + architecture-boundary tests pass

- [x] 6. Platform_SDK: splitters, metrics, time series, engine labelling (design C2)
  - `ml_engine/sdk/__init__.py` (NEW) + submodules: re-export the engine's pure splitters
    (`random/stratified/group/temporal` + an expanding-window/rolling-origin split), the metric
    functions, forecasting metrics (MAE/RMSE/MAPE/sMAPE + seasonal-naive baseline), and
    `io.write_results(..., sdk=True)`; a `SDK_VERSION` constant recorded on the run
  - The temporal/rolling-origin splitters refuse a config that leaks future into past
    (`TEMPORAL_LEAK`, Req 5.4); the job labels SDK-written metrics `engine` (Req 4.2)
  - Tests: a Hypothesis property test that every temporal split is train-before-eval (Property 3);
    a refusal test for a leaking config; an SDK metric equals the managed metric on the same inputs
    (Property 4); the seasonal-naive baseline is always surfaced (Req 5.5)
  - _Requirements: 3.3, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 7. Independent validation + model-version enumeration + comparison (design C5, C7, C9)
  - `backend/services/model_versions.py`: enumerate a free-form run's model (source="free_form")
    alongside the managed models/<name>/ tree so it is listable/validatable (Req 6.3)
  - `backend/services/free_form.py`: `validate_run_model` pass-through to the existing
    ValidationService for the run's Model_Version (engine metrics, Req 7.1, 7.2); a run with no
    Predict_Entry_Point says validation is unavailable (Req 7.4)
  - `backend/api` + `mcp_server`: a comparison surface listing free-form and managed runs together
    with tier + metric labels, like-for-like on engine metrics only, stating when not comparable
    (Req 8); no auto-select (Req 8.3)
  - Manifest update for model_versions.py if hash-tracked; regenerate
  - Tests: a Predict_Entry_Point model validates through the standard flow with engine metrics; a
    run without one reports validation unavailable; the comparison labels every entry (Property 5)
  - _Requirements: 6.3, 7.1, 7.2, 7.4, 8.1, 8.2, 8.3_

- [x] 8. Leakage screen on free-form inputs (design C4; reuse analyze_leakage)
  - Screen the run's dataset/target/declared features with `analyze_leakage` (reusing the managed
    rules + `LeakageConfig` unchanged, Req 13.1) in a Processing step before approval; profile the
    dataset the same way `jobs/profiling/main.py` does to get the EdaReport it needs. No target ->
    only target-free checks, recording which were skipped (Req 13.6); labels-for-analysis ->
    screen features against those labels (Req 13.7). Attach the LeakageReport to the run; advisory,
    never auto-edit (Req 13.2, 13.3); approving with open high-severity findings records the
    acknowledgement (Req 13.4)
  - Tests: the dataset is screened before approval; findings attach and are shown; a finding never
    blocks/edits the run; the unsupervised path runs only target-free checks (Property 7)
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.6, 13.7_

- [x] 9. Data-context Q&A gate + AI drafting (design C5)
  - `backend/services/free_form.py`: `request_data_context` returns the Req-14 question set (seeded
    from the profile summary + column names + saved defaults); `draft_script` refuses until the
    questions are answered or skipped (`DATA_CONTEXT_REQUIRED`), records the answers/skips on the
    run, asks the LLM for a `ScriptRevision` (profile summary + answers only, never rows), stores it
    with a diff, and returns it. The LLM may not approve or submit (B12). The draft states which
    answer drove each split/feature choice and which risks stay open for "don't know"/skips
    (Req 14.4, 14.5). Answers are reused as defaults for the dataset (Req 14.6)
  - `backend/api` + `mcp_server`: surface request-questions / draft / revise, echoing the stored
    answers; audited (Req 14.7)
  - Tests: a draft cannot be produced before the questions are answered/skipped; answers are stored
    and passed to drafting; revisions are retained and diffed (Property 8)
  - _Requirements: 2.1, 2.2, 2.3, 2.7, 14.1, 14.3, 14.4, 14.5, 14.6, 14.7_

- [x] 10. End-to-end, backward-compat and golden tests (design Testing Strategy)
  - End-to-end offline (InlineOrchestrator): draft -> approve -> leakage screen -> run -> labelled
    result -> (optional) validate the run's Model_Version through the standard flow; a per-run
    failure is isolated. Backward compatibility: an experiment with no free-form run is byte-
    identical to today (Property 9, golden)
  - _Requirements: 3.2, 6.2, 8.1, 11.2, 12.1_

- [x] 11. Final checkpoint - Ensure all tests pass
  - Run `.venv\Scripts\python -m pytest -q -p no:cacheprovider`, `.venv\Scripts\python -m ruff check .`,
    `.venv\Scripts\python -m ruff format --check .`; confirm `tests/test_inherited_manifest.py`
    and `tests/test_architecture_boundaries.py` pass; ensure all pass; ask the user if questions arise
  - Result: full suite green (exit 0); ruff check + format clean; manifest + architecture-boundary
    tests pass

## Notes

- A free-form run is a `SIDE_WORKFLOWS` Processing job, so orchestration, status monitoring and the
  stop action are reused; adding FREE_FORM to the frozenset is what makes routing correct.
- The Platform_SDK re-exports the engine's existing pure splitters/metrics (Req 4.3), so an
  SDK metric equals the managed metric by construction — no parallel implementation to drift.
- The script runs with the job's existing IAM role and data access (sandboxing is out of scope for
  trusted users); the platform labels results rather than guaranteeing the script is leak-free.
- Property tests use Hypothesis (`max_examples=100`, `deadline=None`, `derandomize=True`), tagged
  `# Feature: free-form-experiments, Property N`.
