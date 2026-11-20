# Design: Stock-image migration

## Overview

This design turns `decision.md` (B1–B13) and `requirements.md` into concrete code changes. It is
split into **Stage 1** (infra swap, fully offline-testable, lands first) and **Stage 2** (the
per-experiment install layer, coupled to real-AWS behaviour, lands after a first real-AWS run).

The deterministic engine, the workflow progression (`advance`), the guards, tracking, leakage,
reporting and audit are unchanged except for the specific edits named here. The change is mostly in
the control plane (image resolution, orchestrator request building, config) plus three small,
manifest-tracked edits to inherited engine files.

## Architecture

The one production topology is unchanged: control plane submits SageMaker jobs that read/write S3
through `ObjectStore` and run `python -m jobs.<stage>.main`. The only architectural change is where
the **image URI** comes from:

- Before: `settings.stage_image_uri` (an operator-typed ECR URI of our custom image), required.
- After: `backend/aws/images.py::resolve_stage_image(settings)` returns the image URI at
  submission — the configured override if set, else an AWS SDK lookup of the stock XGBoost image
  for the region. The resolved URI is recorded on `ExecutionRecord.image_uri`.

The `sagemaker` SDK is a control-plane-only, lazily-imported dependency; `ml_engine` keeps no AWS
dependency (architecture boundary preserved). Stage 2 adds a per-experiment `requirements` install
layer on top of the same stock image, with the resolved versions captured by `pip freeze` and saved
as an artifact.

## Stage split

| Stage | Scope | Offline-testable | Requirements |
|---|---|---|---|
| 1 | Image resolver + record digest; drop checkpoints/`RESUMABLE_MODELS`; unregister CatBoost; loosen `pyproject`; delete Dockerfile; `resolve_stage_image` MCP tool; LLM-boundary reword | Fully | 1, 4, 5, 6, 7, 8, 9, 10 |
| 2 | Per-experiment `requirements` field + install-at-job-start + saved `pip freeze`; reproducibility record | Contracts + resolver offline; install step confirmed at first real-AWS run | 2, 3 |

Stage 1 leaves the platform working on the stock image with today's model set. Stage 2 reopens the
catalogue. This document specifies both; implementation does Stage 1 first.

---

## Components and Interfaces

Stage 1 components (C1–C10) below; Stage 2 components (D1–D5) follow in their own subsection.

### C1. Image resolver (`backend/aws/images.py`, NEW)

A single control-plane function, AWS SDK imported lazily so `ml_engine` keeps no AWS dependency.

```python
def resolve_stage_image(settings: Settings) -> ResolvedImage:
    """Return the stage image URI for a run and where it came from.

    - settings.stage_image_uri set  -> ResolvedImage(uri=that, source="config")
    - unset -> sagemaker.image_uris.retrieve(framework="xgboost",
               region=settings.aws_region, version=<resolved>) ->
               ResolvedImage(uri=..., source="aws_lookup")
    """
```

- `ResolvedImage` is a small frozen dataclass (not a Pydantic artifact): `uri: str`,
  `source: Literal["config", "aws_lookup"]`.
- The `sagemaker` import is inside the function body (lazy). An `ImportError` or an SDK error
  becomes `ImageResolutionError(code, message, region)` (a plain `RuntimeError` subclass in
  `backend/aws/images.py`), so a missing SDK or unknown region fails with a structured message and
  no job is submitted (Req 1.6).
- A new optional setting `xgboost_image_version: str | None = None` lets an operator pin the
  framework version passed to `image_uris.retrieve`; `None` lets the SDK pick its default. (Open
  question 1.)

### C2. Config changes (`backend/config.py`)

- `stage_image_uri: str = ""` stays, now **optional in meaning**: `problems()` no longer appends
  "ML_FACTORY_STAGE_IMAGE_URI is required". The instance-type checks and everything else are
  unchanged.
- Add `xgboost_image_version: str | None = None`.
- Add the Stage-2 settings now so config is stable (unused until Stage 2):
  `package_install_allowlist: list[str] = []` and `enforce_package_allowlist: bool = False`.
- `problems()` loses only the `stage_image_uri` requirement. Nothing else moves.

### C3. Orchestrator: resolve at submission, record the digest, drop checkpoints

`backend/orchestration/sagemaker.py`:

- **Resolve once per workflow.** In `_open` (or at the top of each `start_*`), call
  `resolve_stage_image(self._settings)` and thread the resolved URI into the execution: set
  `execution.image_uri = resolved.uri` (the field already exists and is already recorded). The
  resolved URI is passed to the request builders instead of `settings.stage_image_uri`.
- **`_image(settings, plan)`** changes from `plan.image_uri or settings.stage_image_uri` to
  `plan.image_uri or <resolved уri threaded in>`. Cleanest: add a `resolved_image: str` parameter
  to the request builders and to `_submit`, or store it on the execution and read it there. The
  side-job path already overrides with `training_image(...)`; that stays — a holdout/ablation/
  scoring job still reuses the recorded training image, which now is the resolved stock image.
- **Checkpoints removed.** In `_training_plan`:
  - `resumable = model in RESUMABLE_MODELS` becomes always `False` (because `RESUMABLE_MODELS` is
    empty — C5), so `checkpoint_uri` is always `None` and the `--checkpoint-dir` argument is never
    added.
  - Leave the `checkpoint_uri` plumbing in `JobPlan`/`SubmittedJob` (it just stays `None`), so the
    change is minimal and reversible.
  - `build_training_request` already guards `if plan.checkpoint_uri:` before adding
    `CheckpointConfig`; with `checkpoint_uri=None` no `CheckpointConfig` is ever set (Req 5.2).
  - Managed Spot is untouched: `use_spot`, `max_wait_seconds`, `EnableManagedSpotTraining` all
    stay.
- `_common_env` sets `ML_FACTORY_STAGE_IMAGE_URI` to the resolved URI so the job records the exact
  image it ran on (unchanged mechanism, new value source).

### C4. Record the image digest on the run

The execution already stores `image_uri`. For reproducibility (Req 3.2) the **digest** must be
recorded on the model-version/run metadata, not only the execution:

- When the resolver returns a URI that already contains a digest (`...@sha256:...`), record it as
  is. When it returns a tag, the training job (which runs on the image) is the natural place to
  resolve the running image's digest, but that needs the container to know its own digest, which
  SageMaker does not always expose. **Decision:** record `execution.image_uri` as the
  reproducibility key for Stage 1; if it is a tag, that is the key. Resolving a tag to a digest is
  a Stage-2/real-AWS refinement (the resolver can be asked to prefer a digest-qualified URI). This
  keeps Stage 1 fully offline.

### C5. Engine: empty `RESUMABLE_MODELS`, unregister CatBoost (inherited, manifest)

- `ml_engine/contracts/execution.py`: `RESUMABLE_MODELS = frozenset()`. Keep the constant and its
  docstring (now "no model resumes; every model is restart-on-interruption"). This is an inherited
  file -> manifest update + justification.
- `ml_engine/models/registry.py`: remove `CatBoostClassifierPlugin` / `CatBoostRegressorPlugin`
  from the import and the registration loop. Leave `ml_engine/models/catboost_plugin.py` on disk
  (unimported) so it can be re-registered via the open catalogue later, OR delete it; **decision:**
  keep the file, drop only the registration, so Stage 2 / the ensembles spec can register it on
  demand when `catboost` is installed. Inherited file -> manifest update + justification.

### C6. Training-stage checkpoint module becomes inert

`jobs/training/checkpointing.py`: with `RESUMABLE_MODELS` empty, `checkpointing_factory` (imported
by the training job) must still import and return a no-op. Keep the module and its public symbol;
make the factory return a checkpointer that does nothing (no read, no write). The CV fold-level
per-fold persistence in `jobs/training/cv.py` is independent and untouched (Req 5.4).

### C7. `pyproject.toml` and build artifacts

- Replace `==` pins with loose lower bounds where an API is needed (`pandas>=2`,
  `scikit-learn>=1.3`, `pydantic>=2`, …) or no bound; `requires-python` a loose floor. Drop
  `catboost` from the `models` extra (XGBoost stays). (Req 6.1.)
- Add `sagemaker` to the `backend` extra for `image_uris` (Req 6.2). Lazy-imported in
  `backend/aws/images.py`; never imported from `ml_engine`.
- Delete `container/Dockerfile` and `.dockerignore` (Open question 2), and drop any manifest/test
  entry that referenced them.

### C8. MCP `resolve_stage_image` tool (`mcp_server/server.py`)

- Add a read-only tool `resolve_stage_image()` returning `{uri, source}` from
  `resolve_stage_image(settings)`. Annotate `readOnlyHint=True`; it performs no AWS mutation and
  starts nothing (Req 7).
- The in-process FastMCP client test stubs the resolver so no AWS is needed; update the
  tool-discovery test's expected set to include the new tool.

### C9. LLM-boundary reword (docs/wording only)

- Replace "the LLM never decides infrastructure" wording in the `ml-factory-2` design and any spec
  clause with the B12 form: propose and explain, compute no number, apply nothing without explicit
  user approval. No code behaviour changes (the approval gating already exists). (Req 8.)
- Update `ml-factory-2/design.md` conflicts: Q5 (ECR required), DD-7 (one custom image), the
  reproducibility property, the model list (CatBoost), Req 4.3 (checkpoint resume -> not
  applicable). (Req 9.4.)

### C10. Tests (Stage 1)

- `backend/aws/images.py`: explicit URI passes through (`source="config"`); unset calls a stubbed
  `image_uris.retrieve` with `region=settings.aws_region` and the configured version
  (`source="aws_lookup"`); a stubbed SDK error raises `ImageResolutionError` with the region.
- Orchestrator: with a stubbed resolver, `start_training` records `execution.image_uri`; **no job
  sets `CheckpointConfig`** (assert on the built request dict for a would-be-resumable model like
  `xgboost`); `RESUMABLE_MODELS == frozenset()`.
- Registry: `catboost` / `catboost_regressor` not in `all_plugins()` / `default_model_names`; a
  plugin whose library is absent reports `is_available() == (False, reason)` without raising.
- MCP: `resolve_stage_image` via the in-process client with a stubbed resolver; tool-discovery set
  updated.
- Remove/repurpose the static-Dockerfile test (`tests/infrastructure/test_stage_image.py` if
  present); relax any reproducibility test from version-match to digest/URI.
- Manifest: regenerate after the `execution.py` and `registry.py` edits; `test_inherited_manifest`
  green.

### Stage 2 components (planned; build after a first real-AWS run)

#### D1. `requirements` on the request and frozen config

- `backend/schemas.py::TrainingRequest`: add `requirements: list[str] = []` (the experiment's pip
  list). Optional; absent reproduces today.
- `ml_engine/contracts/config.py`: a new versioned contract `RequirementsSpec` (or a field on the
  frozen config) carrying the approved list, `exclude_if` empty so single-image configs stay
  byte-identical. Validate each entry is a well-formed requirement specifier before submission
  (Req 2.4); if `enforce_package_allowlist`, refuse entries outside `package_install_allowlist`
  (Req 2.5).

#### D2. Install at job start

- The orchestrator passes the approved `requirements` to the job via the stock framework image's
  dependency mechanism (a `requirements.txt` placed where the SageMaker framework container
  installs it before the entrypoint, or the SDK `dependencies`/`source_dir` convention). Exact
  mechanism confirmed at the first real-AWS run (decision open item). The corporate proxy is
  already in `Settings.https_proxy` / `backend/aws/clients.py`.
- A failed install fails the job with a structured error and captured logs; no fallback (Req 2.6).

#### D3. Resolved_Requirements (`pip freeze`) saved per experiment

- New layout path in `ml_engine/io/layout.py`, e.g. `requirements_resolved` ->
  `config/requirements.resolved.txt`.
- After install resolves, the job runs `pip freeze` and writes it there. A new contract or a plain
  text artifact records it; the model-version metadata references it alongside the image digest
  (Req 3.1, 3.2). Empty-requirements runs still write the image's frozen set (Open question 4).

#### D4. Reproducibility record

- Reproducibility key becomes (image digest + Resolved_Requirements) (Req 3.3). The offline suite
  is not required to match the image (Req 3.4).

#### D5. Tests (Stage 2)

- Contract round-trip and backward compatibility for `requirements` (absent == today).
- Well-formedness validation rejects a malformed specifier; allowlist (when enabled) rejects an
  outside package, submits no job.
- Install-step wiring tested with a fake SageMaker (the request carries the requirements); the
  actual install is a first-real-AWS confirmation, not an offline test.
- `pip freeze` capture round-trips to the store (job-level test with a stubbed freeze).

## Data Models

New and changed contracts/types (paths defined only where noted):

- `ResolvedImage` (frozen dataclass, `backend/aws/images.py`, NOT an artifact): `uri: str`,
  `source: Literal["config", "aws_lookup"]`.
- `ImageResolutionError(RuntimeError)`: `code: str`, `message: str`, `region: str`.
- `Settings` (changed, `backend/config.py`): `stage_image_uri` no longer required by `problems()`;
  new `xgboost_image_version: str | None`, `package_install_allowlist: list[str]`,
  `enforce_package_allowlist: bool`.
- `RESUMABLE_MODELS` (changed, `ml_engine/contracts/execution.py`): now `frozenset()`.
- `ExecutionRecord.image_uri` (existing field, now populated from the resolver).
- Stage 2: `TrainingRequest.requirements: list[str]` (default `[]`); a versioned `RequirementsSpec`
  or frozen-config field (`exclude_if` empty); a new layout path `requirements.resolved.txt` in
  `ml_engine/io/layout.py` for the `pip freeze` artifact.

No existing artifact shape changes in Stage 1; single-split/CV artifacts stay byte-identical.

## Correctness Properties

### Property 1: resolver determinism
For a given `Settings`, `resolve_stage_image` returns the same URI and source; an explicit
`stage_image_uri` always passes through unchanged.
**Validates: Requirements 1.1, 1.2, 1.3**

### Property 2: no checkpoint
No submitted job carries a `CheckpointConfig`, for any model, and `RESUMABLE_MODELS` is empty.
**Validates: Requirements 5.1, 5.2**

### Property 3: image recorded
Every submitted training execution records a non-empty `ExecutionRecord.image_uri` equal to the
resolver's output.
**Validates: Requirements 1.4**

### Property 4: catalogue
CatBoost is not registered by default; a model whose library is absent reports unavailable instead
of raising.
**Validates: Requirements 4.1, 4.2, 4.3**

### Property 5: backward compatibility
A run with no `requirements` (Stage 2) behaves exactly as a Stage 1 run; config JSON omits the field
at its default.
**Validates: Requirements 3.5, 9.1**

## Error Handling

- Resolver: missing SDK or unknown region -> `ImageResolutionError(code, message, region)` before
  any submission; the workflow is not opened.
- Install (Stage 2): a failed `pip install` fails the job with a structured error and captured
  logs; never a silent fallback to a different package set.
- Allowlist (Stage 2, when enabled): a package outside the allowlist is refused in the service
  layer before submission, so no partial job runs.

## Testing Strategy

All offline with `InMemoryObjectStore`, an in-process orchestrator and a fake SageMaker; the AWS
SDK `image_uris.retrieve` is stubbed. Stage 1 tests (C10) cover the resolver, the no-checkpoint
property, the recorded image, the registry change and the MCP tool. Stage 2 contract tests (D5) are
offline; the actual install and `pip freeze` on the real image are first-real-AWS confirmations.

## What changes in inherited files (manifest)

| File | Change | Justification |
|---|---|---|
| `ml_engine/contracts/execution.py` | `RESUMABLE_MODELS = frozenset()` | Decision B4: no model resumes; Spot is restart-on-interruption. Constant kept for imports. |
| `ml_engine/models/registry.py` | Drop CatBoost from the registration loop | Decision B2/B10: CatBoost moves to the install-driven open catalogue; base image has sklearn + XGBoost. |
| `jobs/training/checkpointing.py` | `checkpointing_factory` returns a no-op | Decision B4: checkpoint/resume dropped; module kept so the training job's import resolves. |

Each edit updates `scripts/inherited_manifest.py` (move to `ADAPTED` with the justification) and
regenerates `tests/inherited_manifest.json` in the same task.

## Risks and first-run confirmations (from decision.md)

1. The AWS XGBoost image includes scikit-learn at a version the sklearn plugins run on — confirmed
   by running one sklearn and one XGBoost model on the image once. If false, Option B needs two
   images (reopens DD-7).
2. The image accepts a plain `ContainerEntrypoint` for Training jobs (works for Processing).
3. Our code installs cleanly on top with loose pins (no resolver conflict against the image's
   preinstalled versions).

None block Stage 1; all are relevant to Stage 2's install step.
