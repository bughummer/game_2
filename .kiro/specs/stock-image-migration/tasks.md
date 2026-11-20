# Implementation Plan: Stock-image migration

## Overview

Python, following `design.md`. **Stage 1** (tasks 1–9) is built now and is fully offline-testable.
**Stage 2** (tasks 10–13) is planned but deferred until a first real-AWS run confirms the install
step; its tasks are listed so the plan is complete. Commands: `.venv\Scripts\python -m pytest -q`,
`.venv\Scripts\python -m ruff check .`, `.venv\Scripts\python -m ruff format --check .`. Every
inherited-file change updates `scripts/inherited_manifest.py` and regenerates
`tests/inherited_manifest.json` in the same task.

## Task Dependency Graph

```
1 image resolver + config
└─▶ 2 orchestrator: resolve + record + drop checkpoints
     ├─▶ 3 engine: empty RESUMABLE_MODELS (inherited)
     ├─▶ 4 engine: unregister CatBoost (inherited)
     ├─▶ 5 training checkpoint module inert (inherited)
     └─▶ 6 pyproject + delete Dockerfile
          └─▶ 7 MCP resolve_stage_image tool
               └─▶ 8 LLM-boundary reword + ml-factory-2 doc updates
                    └─▶ 9 Stage 1 checkpoint (suite + ruff + manifest)

Stage 2 (deferred):
9 └─▶ 10 requirements contract + request field
         └─▶ 11 install-at-job-start wiring
              └─▶ 12 pip freeze capture + reproducibility record
                   └─▶ 13 Stage 2 checkpoint
```

```json
{
  "waves": [
    { "wave": 1, "tasks": ["1"], "depends_on": [] },
    { "wave": 2, "tasks": ["2"], "depends_on": ["1"] },
    { "wave": 3, "tasks": ["3", "4", "5", "6"], "depends_on": ["2"] },
    { "wave": 4, "tasks": ["7"], "depends_on": ["6"] },
    { "wave": 5, "tasks": ["8"], "depends_on": ["7"] },
    { "wave": 6, "tasks": ["9"], "depends_on": ["3", "4", "5", "8"] },
    { "wave": 7, "tasks": ["10"], "depends_on": ["9"] },
    { "wave": 8, "tasks": ["11"], "depends_on": ["10"] },
    { "wave": 9, "tasks": ["12"], "depends_on": ["11"] },
    { "wave": 10, "tasks": ["13"], "depends_on": ["12"] }
  ]
}
```

## Tasks

- [x] 1. Image resolver and config (design C1, C2)
  - Create `backend/aws/images.py` with `ResolvedImage` (frozen dataclass: `uri`, `source`),
    `ImageResolutionError(RuntimeError)` (with `code`, `message`, `region`), and
    `resolve_stage_image(settings) -> ResolvedImage`: returns the configured `stage_image_uri`
    (`source="config"`) when set, else lazily imports `sagemaker` and calls
    `image_uris.retrieve(framework="xgboost", region=settings.aws_region, version=settings.xgboost_image_version)`
    (`source="aws_lookup"`); an ImportError or SDK error raises `ImageResolutionError` naming the region
  - `backend/config.py`: drop the `stage_image_uri` requirement from `problems()`; add
    `xgboost_image_version: str | None = None`, `package_install_allowlist: list[str] = []`,
    `enforce_package_allowlist: bool = False`
  - Tests: explicit URI passthrough; unset -> stubbed `image_uris.retrieve` called with the region
    and version; stubbed SDK error -> `ImageResolutionError`; `problems()` no longer flags a missing
    image when the rest is valid
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6_

- [x] 2. Orchestrator: resolve at submission, record, drop checkpoints (design C3, C4)
  - In `backend/orchestration/sagemaker.py`, resolve the image once per workflow in `_open`/`start_*`
    and thread the resolved URI into the request builders (replace `settings.stage_image_uri` in
    `_image`/`_common_env`); set `execution.image_uri` to the resolved URI
  - `_training_plan`: `checkpoint_uri` always `None` (do not add `--checkpoint-dir`); leave the
    `JobPlan`/`SubmittedJob` fields in place; confirm `build_training_request` sets no
    `CheckpointConfig`; Managed Spot unchanged
  - Tests (fake SageMaker + stubbed resolver): `start_training` records `execution.image_uri`; the
    built training request for `xgboost` has no `CheckpointConfig`; Spot fields still present; side
    jobs still reuse `training_image(...)`
  - _Requirements: 1.4, 5.2, 5.3_

- [x] 3. Empty `RESUMABLE_MODELS` (inherited; design C5)
  - `ml_engine/contracts/execution.py`: `RESUMABLE_MODELS = frozenset()`, docstring updated; keep
    the constant
  - Move the file to `ADAPTED` in `scripts/inherited_manifest.py` with a justification; regenerate
    `tests/inherited_manifest.json`
  - Tests: `RESUMABLE_MODELS == frozenset()`; invert any existing "only resumable models get a
    checkpoint" test to "no job sets a checkpoint"
  - _Requirements: 5.1, 9.3_

- [x] 4. Unregister CatBoost (inherited; design C5)
  - `ml_engine/models/registry.py`: remove the two CatBoost plugins from the import and the
    registration loop; keep `ml_engine/models/catboost_plugin.py` on disk (unimported)
  - Manifest update + justification; regenerate
  - Tests: `catboost`/`catboost_regressor` absent from `all_plugins()` and `default_model_names`;
    a plugin with an absent library reports `is_available() == (False, reason)` without raising
  - _Requirements: 4.1, 4.2, 4.3, 4.5, 9.3_

- [x] 5. Training checkpoint module inert (inherited; design C6)
  - `jobs/training/checkpointing.py`: `checkpointing_factory` returns a no-op checkpointer (no read,
    no write); keep the public symbol so the training job imports resolve
  - Manifest update + justification if the file is inherited; regenerate
  - Tests: the factory returns a no-op; a training run with a would-be-resumable model writes no
    checkpoint; CV fold persistence in `jobs/training/cv.py` still works (existing CV tests pass)
  - _Requirements: 5.4_

- [x] 6. `pyproject.toml` and build artifacts (design C7)
  - Replace `==` pins with loose lower bounds / none; `requires-python` a loose floor; drop
    `catboost` from the `models` extra; add `sagemaker` to the `backend` extra
  - Delete `container/Dockerfile` and `.dockerignore`; remove any manifest/test entry referencing
    them
  - Reinstall into `.venv` and confirm the suite still imports/resolves
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 7. MCP `resolve_stage_image` tool (design C8)
  - `mcp_server/server.py`: add a read-only `resolve_stage_image()` tool returning `{uri, source}`
    via `resolve_stage_image(settings)`; `readOnlyHint=True`; no mutation
  - Tests: in-process FastMCP client with a stubbed resolver returns the URI and source; update the
    tool-discovery expected set to include it
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 8. LLM-boundary reword and ml-factory-2 doc updates (design C9)
  - Replace "the LLM never decides infrastructure" wording with the B12 form (propose and explain,
    compute no number, apply nothing without approval) in `.kiro/specs/ml-factory-2/design.md` and
    any remaining spec clause
  - Update `ml-factory-2/design.md` conflicts: Q5 (ECR required), DD-7 (one custom image), the
    reproducibility property, the model list (CatBoost), Req 4.3 (checkpoint resume -> not
    applicable)
  - Docs/wording only; no code behaviour change
  - _Requirements: 8.1, 8.2, 8.3, 9.4_

- [x] 9. Stage 1 checkpoint - Ensure all tests pass
  - Run pytest, `ruff check` and `ruff format --check`; confirm `tests/test_inherited_manifest.py`
    passes with the regenerated manifest; the former static-Dockerfile test is removed/repurposed;
    ensure all tests pass, ask the user if questions arise
  - Result: 1283 passed, 1 skipped; ruff check + format clean; inherited manifest test passes
    (execution.py, registry.py, checkpointing.py recorded as ADAPTED with justifications); the
    static Dockerfile test and `container/Dockerfile` + `.dockerignore` were removed
  - _Requirements: 10.1, 10.2, 10.3, 10.5, 10.6, 10.7_

- [x] 10. (Stage 2) `requirements` contract + request field (design D1) — pulled forward
  - `backend/schemas.py::TrainingRequest.requirements: list[str] = []`; new versioned contract
    `ml_engine/contracts/requirements.py::RequirementsSpec` with per-line well-formedness
    (`validate_requirement_line`) and `requirement_name`; `ExperimentConfig.requirements`
    (`RequirementsSpec | None`, excluded at the None default); `ExperimentService._resolve_requirements`
    validates before archiving and enforces the allowlist when `enforce_package_allowlist`
  - Tests in `tests/backend/test_requirements_spec.py`: well-formed accepted / malformed rejected;
    duplicates collapsed; round-trip; empty omitted from config JSON (byte-identical); frozen into
    config; malformed refused with nothing submitted; allowlist blocks an outside package when on
  - Manifest: `contracts/requirements.py` added as NEW; `config.py` justification extended;
    regenerated
  - _Requirements: 2.1, 2.3, 2.4, 2.5, 9.1, 9.2_

- [ ] 11. (Stage 2, deferred) Install at job start (design D2)
  - Orchestrator passes the approved `requirements` via the stock framework image's dependency
    mechanism before the entrypoint; proxy honoured; failed install fails the job with a structured
    error and no fallback
  - Tests: the built request carries the requirements (fake SageMaker); install failure path (the
    real install is a first-real-AWS confirmation)
  - _Requirements: 2.2, 2.6_

- [ ] 12. (Stage 2, deferred) `pip freeze` capture + reproducibility (design D3, D4)
  - New layout path for `requirements.resolved.txt`; the job writes `pip freeze` after install;
    model-version metadata references it with the image digest; empty-requirements still records a
    set
  - Tests: capture round-trips to the store (stubbed freeze); reproducibility record has digest +
    resolved requirements
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 13. (Stage 2, deferred) Stage 2 checkpoint
  - Full suite + ruff + manifest green; confirm backward compatibility (no-requirements run
    byte-compatible with Stage 1 behaviour)
  - _Requirements: 10.1, 10.4_

## Notes

- Stage 2 (tasks 10–13) is intentionally not started until a first real-AWS run confirms the three
  decision open items (image ships sklearn; plain entrypoint accepted for Training; our code
  installs cleanly). Building the contracts (task 10) early is safe and may be pulled forward if
  wanted.
- Every inherited-file task (3, 4, 5) updates the manifest in the same task; task 9 verifies it.
