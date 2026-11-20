# Phase 0 — Assessment of the reference repository

ML Factory 2.0 is a new project. `../old_repo` is the reference and the source of the proven
deterministic engine. Everything below was verified against `old_repo` on 2026-10-14.

## Verified facts

### Test baseline of what we inherit
`python -m pytest tests/ml_engine tests/jobs tests/test_architecture_boundaries.py` in `old_repo`
→ **490 passed, 0 failed** (Python 3.14.5 locally; `pyproject.toml` requires `>=3.12`).
The full old suite (652, incl. backend/workflow) is not our baseline: those layers are rebuilt.

### The deleted AWS code is NOT recoverable
`old_repo` is not a git repository (no `.git`). The handover's "recover `ml_engine/io/s3.py` and
the `StepFunctionsOrchestrator` from git history" is impossible. A repo-wide search finds no
surviving `S3ObjectStore` or Step Functions orchestrator. **Consequence:** both are written fresh
against the current Protocols (DD-2 below). Low risk: `ObjectStore` is 8 methods.

`ml_engine/io/layout.py` still documents `s3://bucket/ml-factory/experiments/<id>` as the base,
and `backend/schemas/experiments.py` still describes `dataset_uri` as an S3 URI — the layout was
designed for S3 and needs no change.

### The seams
- `ml_engine/io/object_store.py` — `ObjectStore` Protocol: `read_bytes`, `write_bytes`, `exists`,
  `list_uris`, `list_prefixes`, `stat`, `download`, `upload`; plus `ObjectMetadata`,
  `ObjectNotFoundError`, and helpers `write_model`/`read_model`/`read_model_if_exists`/
  `write_json`/`copy_object`. `LocalObjectStore` lives in the same file and is **not** ported.
- `ml_engine/io/uri.py` — `file://`-only today. Backend-agnostic helpers to keep: `join_uri`,
  `uri_matches_prefix`, `detect_format`. File-specific helpers to drop: `to_file_uri`,
  `file_uri_to_path`, `normalize_storage_root` (rewritten for `s3://`), `InvalidFileUriError`.
- `ml_engine/io/dataset.py` — all I/O already goes through `store.download/upload/stat/list_uris`
  into temp dirs. Works unchanged on S3.
- `backend/orchestration/base.py` — `ExperimentOrchestrator` Protocol (`name`, `start_eda`,
  `start_training`, `describe`) and `ExecutionHandle`. Depends only on
  `ml_engine.contracts.experiment.ExperimentRecord`. Ported verbatim.
- `jobs/_common/runtime.py::build_store()` hard-returns `LocalObjectStore()`; `base_parser` help
  text mentions `file://`. Both change; `run_entrypoint` and failure-artifact logic are kept.

### Test dependencies on the local backend
- `tests/conftest.py` builds `LocalObjectStore`, `tmp_path`-based `file://` roots, `Settings`
  with `queue_dir`, and `backend.container` with `InlineOrchestrator`. It must be rewritten.
- `tests/ml_engine/test_io.py` has ~12 tests specifically about `file://`/`LocalObjectStore`;
  those are dropped and replaced by S3 tests. The remaining engine tests take a `store` fixture
  and are expected to run unchanged against a test-only `InMemoryObjectStore`.

### Prior art in `old_repo/aws-canvas-master/`
Standalone SageMaker Canvas scripts: Processing/Training/Batch Transform submission, a cost
poller, budget and Bedrock kill-switch policies, a SageMaker job EventBridge pattern. Reference
only (hardcoded names, print-based); not copied.

## Inherit / drop / build

| Area | Action |
|---|---|
| `ml_engine/contracts, profiling, leakage, features, preprocessing, splitting, models, evaluation, reporting, reasoning, training, dictionary` | copy verbatim |
| `ml_engine/io/object_store.py` | copy minus `LocalObjectStore` |
| `ml_engine/io/uri.py` | rewrite for `s3://` only, keep backend-agnostic helpers |
| `ml_engine/io/s3.py` | **write new** `S3ObjectStore` (client injected) |
| `ml_engine/io/layout.py`, `dataset.py` | copy verbatim; layout gains per-run prefixes later |
| `jobs/*` | copy; `build_store` returns `S3ObjectStore` |
| `backend/orchestration/base.py` | copy verbatim |
| `backend/*` (services, api, schemas, repositories) | port selectively, module by module, dropping queue/local |
| engine + jobs tests | copy; swap fixtures to `InMemoryObjectStore`; drop `file://` tests |
| `workflow/`, `LocalObjectStore`, `file://`, queue settings, two-container compose | do not port |
| `frontend/` | not ported in v1 (open question Q6) |

## Risks
- **Silent drift from the engine.** Any "improvement" to copied engine code breaks the trust
  baseline. Mitigation: copy byte-for-byte, then a test asserts the copied files hash-match a
  recorded manifest until a deliberate change is reviewed.
- **Test-only store diverging from S3 semantics** (prefix listing, trailing slashes, missing
  objects). Mitigation: one shared contract-test suite runs against both `InMemoryObjectStore` and
  `moto`-backed `S3ObjectStore`.
- **Reproducibility across SageMaker runs.** Same container image + frozen config + fixed seeds
  should match; thread-count nondeterminism (XGBoost/CatBoost) is the known risk. Pin
  `nthread`/`thread_count` in the container env if the reproducibility test fails.
- **Concurrent writes** from parallel per-model jobs — per-run records + one aggregation writer.
- **Spot resumability** — only XGBoost and CatBoost checkpoint; sklearn models restart.
