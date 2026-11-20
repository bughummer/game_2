# Requirements Document

Run every SageMaker job on one AWS-prebuilt image instead of a custom image we build and push,
resolve that image automatically, allow each experiment to install its own user-approved packages
at job start, record the resolved packages for reproducibility, and reword the LLM boundary to
"propose and wait for approval". This turns the decision in `decision.md` (B1–B13) into a buildable,
testable spec.

## Introduction

Today the platform builds a custom stage image, pushes it to ECR, and pins library versions in
`pyproject.toml`; `stage_image_uri` is required and typed by an operator. The owner has decided
(see `decision.md`) to stop doing this. Instead:

- Every Processing and Training job runs the **AWS SageMaker XGBoost prebuilt image** (which also
  ships scikit-learn, pandas, numpy, scipy). Nothing is built or pushed by us.
- The control plane **resolves the image URI itself** at submission from the AWS SDK; an operator
  sets only the region. An explicit override still works, for pinning a digest.
- An experiment may carry a user-authored **`requirements`** list — its own `requirements.txt` —
  that is `pip install`ed on top of the image before the entrypoint. This reopens the model
  catalogue to anything on PyPI (CatBoost, LightGBM, …) with no custom image.
- **Reproducibility** = the recorded image digest plus the exact installed versions
  (`pip freeze`) saved per experiment. Library versions are not pre-pinned.
- **Spot checkpoint/resume is dropped**: every model is restart-on-interruption; Managed Spot (the
  discount) stays.
- The **LLM boundary is reworded platform-wide**: the LLM may propose any choice (packages, image,
  treatment, models) but applies nothing without explicit user approval and computes no number.

This is primarily an infrastructure and configuration change. The deterministic engine, the
orchestrator's workflow, guards, tracking, MCP server, leakage investigation, reporting and audit
are unchanged except for the specific edits below.

## Glossary

- **Platform**: ML Factory 2.0 as a whole.
- **Base_Image**: the one AWS-prebuilt image all jobs run on (the SageMaker XGBoost image, which
  includes scikit-learn). Named by its ECR URI, identified over time by its digest.
- **Image_Resolver**: the control-plane function that returns the image URI for a run — the
  configured override if set, else an AWS SDK lookup for the Base_Image in the region.
- **Requirements**: an experiment's user-authored list of pip packages (its `requirements.txt`),
  installed on top of the Base_Image at job start. Empty means nothing is installed.
- **Resolved_Requirements**: the exact installed versions captured by `pip freeze` after the
  install resolves, saved as an experiment artifact.
- **Restart_On_Interruption**: a Spot-interrupted job restarts from scratch; no checkpoint resume.
- **LLM_Boundary**: the platform rule — the LLM proposes and explains, computes no number, and
  applies nothing without explicit user approval (decision B12).

## Requirements

### Requirement 1: Resolve the image automatically; stop requiring a hand-typed URI

**User Story:** As an operator, I want the platform to find the right AWS image itself, so that I
never manage an image URI by hand.

#### Acceptance Criteria

1. THE `stage_image_uri` setting SHALL become optional; `Settings.problems()` SHALL NOT require it.
2. WHEN `stage_image_uri` is unset, THE Image_Resolver SHALL return the AWS SageMaker XGBoost
   prebuilt image URI for `aws_region` via the SageMaker SDK (`image_uris.retrieve`), lazily
   importing the SDK so `ml_engine` keeps no AWS dependency.
3. WHEN `stage_image_uri` is set, THE Image_Resolver SHALL return it verbatim (an explicit override
   for pinning a specific digest).
4. THE orchestrator SHALL call the Image_Resolver at submission and SHALL record the resolved image
   URI on the `ExecutionRecord` (the `image_uri` field already exists).
5. THE Image_Resolver SHALL be a deterministic lookup; THE LLM SHALL NOT author the URI (it MAY
   surface it read-only, Requirement 7).
6. IF the SDK lookup fails (no SDK, unknown region), THEN THE resolver SHALL raise a structured
   error naming the region and the cause, and no job SHALL be submitted.

### Requirement 2: Install user-approved packages per experiment

**User Story:** As a data scientist, I want each experiment to install the libraries it needs, so
that I can use any model without a custom image.

#### Acceptance Criteria

1. THE training request / frozen configuration SHALL carry an optional user-authored Requirements
   list; its absence SHALL reproduce today's behaviour (nothing installed).
2. AT job start, on the Base_Image, THE Platform SHALL install exactly the Requirements before the
   `python -m jobs.<stage>.main` entrypoint runs, using the stock image's supported dependency
   mechanism (a `requirements.txt` / dependencies list), with the corporate proxy honoured.
3. THE Requirements SHALL be a user input under the LLM_Boundary; THE LLM MAY propose packages but
   SHALL NOT author the approved list or install anything.
4. THE install SHALL be open by default (trusted users): no allowlist/denylist and no package-count
   cap are enforced; only well-formed package names/versions are validated before submission, and
   the existing time/budget guards bound cost.
5. AN allowlist SHALL exist in configuration but be OFF by default; when an operator enables it,
   packages outside it SHALL be refused before submission.
6. IF the install fails, THEN THE job SHALL fail with a structured error and captured logs, and
   SHALL NOT fall back to a different package set.

### Requirement 3: Reproducibility by recorded image digest plus saved requirements

**User Story:** As a user, I want every experiment reproducible, so that I can rerun or validate a
saved model later on the same stack.

#### Acceptance Criteria

1. AFTER the install resolves, THE job SHALL capture the exact installed versions (`pip freeze`)
   and save them as Resolved_Requirements at a path defined only in `ml_engine/io/layout.py`.
2. THE Platform SHALL record, per run, the resolved Base_Image digest and the Resolved_Requirements
   reference on the run / model-version metadata.
3. THE reproducibility guarantee SHALL be "same image digest + same Resolved_Requirements",
   replacing the former "identical pinned versions verified against the offline suite".
4. THE offline test suite SHALL run on whatever versions resolve locally and SHALL NOT be required
   to match the image's versions.
5. WHEN an experiment has empty Requirements, Resolved_Requirements MAY be empty or the image's own
   frozen set; either way the run SHALL remain reproducible from the digest.

### Requirement 4: Drop the custom model set floor; keep XGBoost; CatBoost via requirements

**User Story:** As a maintainer, I want the built-in model set to match what the Base_Image ships,
with everything else reachable via Requirements.

#### Acceptance Criteria

1. THE built-in registered models SHALL be those the Base_Image runs without an install: the
   scikit-learn plugins and XGBoost (logistic_regression, elastic_net, random_forest, xgboost, and
   their regressor variants).
2. THE CatBoost plugins SHALL NOT be registered by default; CatBoost SHALL be reachable by adding
   `catboost` to an experiment's Requirements (the open catalogue, ensembles-and-learners spec).
3. THE plugin `is_available()` mechanism SHALL report a model whose library is not installed as
   unavailable with a clear reason, rather than raising on import.
4. XGBoost SHALL run at whatever version the Base_Image ships unless an experiment's Requirements
   pin a different version.
5. WHERE `ml_engine/models/registry.py` changes (unregistering CatBoost), THE change SHALL be
   additive/subtractive only in the registration loop, recorded in `tests/inherited_manifest.json`
   with a written justification.

### Requirement 5: Drop Spot checkpoint/resume; keep Managed Spot

**User Story:** As a budget owner, I want the Spot discount without checkpoint machinery.

#### Acceptance Criteria

1. `RESUMABLE_MODELS` SHALL become empty; THE constant SHALL remain so dependent imports resolve.
2. THE orchestrator SHALL NOT set a `CheckpointConfig` on any job and SHALL NOT scope a checkpoint
   path; `SubmittedJob.checkpoint_uri` SHALL be `None` and `resumable` SHALL be `False` for every
   job.
3. Managed Spot (the discount) SHALL be unchanged: `spot_enabled` still applies and a Spot
   interruption is Restart_On_Interruption.
4. THE training-stage checkpoint module SHALL become inert (a no-op) rather than be deleted, so the
   training job's imports still resolve; the CV fold-level per-fold persistence SHALL be unaffected
   (it is artifact-level, not native checkpointing).

### Requirement 6: Loosen version pinning; drop the build artifacts

**User Story:** As the owner, I do not want library versions fixed to any current set.

#### Acceptance Criteria

1. `pyproject.toml` SHALL use loose lower-bound ranges only where the code needs a specific API
   (e.g. `pandas>=2`, `scikit-learn>=1.3`), or no bound; `requires-python` SHALL be a loose floor.
   No `==` pins for the ML libraries. `catboost` SHALL be dropped as a hard dependency.
2. THE `sagemaker` SDK SHALL be added to the backend extra (for `image_uris`), lazily imported in
   the control plane, never imported from `ml_engine`.
3. THE `container/Dockerfile` and `.dockerignore` SHALL be deleted or kept only as a clearly-marked
   developer convenience with a note that production uses the Base_Image.
4. THE job IAM role SHALL retain ECR pull permission (the pull is from AWS's registry; the
   permission shape is unchanged).

### Requirement 7: Read-only image resolution tool for the assistant

**User Story:** As an assistant user, I want to ask which image a run would use, without the
assistant being able to change it.

#### Acceptance Criteria

1. THE MCP server SHALL expose a read-only `resolve_stage_image` tool returning the resolved image
   URI and whether it came from config or the AWS lookup, via the same Image_Resolver.
2. THE tool SHALL be annotated read-only, SHALL perform no AWS mutation, and SHALL start nothing.
3. Training SHALL continue to go only through `start_training`, which resolves the image itself and
   runs the guards; the tool SHALL NOT be an input path to a job.
4. THE tool-discovery test's expected tool set SHALL be updated to include it.

### Requirement 8: Reword the LLM boundary platform-wide

**User Story:** As the owner, I want the LLM to propose freely and wait for my approval, instead of
being forbidden to touch infrastructure.

#### Acceptance Criteria

1. THE former rule "the LLM never decides infrastructure" SHALL be replaced wherever it appears
   (design docs, spec clauses, narration) with the LLM_Boundary: propose and explain, compute no
   number, apply nothing without explicit user approval.
2. THE two absolute invariants SHALL remain: the LLM computes no number (every number from the
   deterministic engine or tracking), and nothing it proposes takes effect without user approval,
   recorded in the audit trail.
3. THE reword SHALL be documentation/wording only in this spec; the enforcement (approval-gated
   service calls) already exists and SHALL be unchanged.

### Requirement 9: Contracts, paths and backward compatibility

**User Story:** As a maintainer, I want these changes additive and reversible.

#### Acceptance Criteria

1. THE Requirements and Resolved_Requirements SHALL be versioned Pydantic contracts; the request
   field SHALL be optional with an absent-equals-today default.
2. THE Platform SHALL define every new artifact path only in `ml_engine/io/layout.py`.
3. WHERE an inherited engine file changes, THE change SHALL be additive and recorded in
   `tests/inherited_manifest.json` with a written justification, regenerated via
   `scripts/inherited_manifest.py --write`.
4. THE `ml-factory-2` design SHALL be updated where it conflicts (Q5 ECR-required, DD-7 one custom
   image, Req 3.6/Property 5 reproducibility, the model list, Req 4.3 checkpoint resume).

### Requirement 10: Offline, deterministic testing

**User Story:** As a developer, I want these changes covered offline with no AWS.

#### Acceptance Criteria

1. THE suite SHALL run with `InMemoryObjectStore`, an in-process orchestrator and mocked AWS; the
   SDK `image_uris.retrieve` SHALL be stubbed so no test needs real AWS.
2. THE suite SHALL test the Image_Resolver: an explicit URI passes through; unset calls the stubbed
   lookup with the configured region; a lookup failure raises the structured error.
3. THE suite SHALL test that no job sets a `CheckpointConfig` and that `RESUMABLE_MODELS` is empty
   (inverting the former "only resumable models get a checkpoint" test).
4. THE suite SHALL test the reproducibility record: a run saves Resolved_Requirements and records
   the digest; an empty-Requirements run still records a digest.
5. THE suite SHALL test that CatBoost is not registered by default and that a model with a missing
   library reports unavailable rather than raising.
6. THE suite SHALL test the `resolve_stage_image` MCP tool via the in-process FastMCP client with a
   stubbed resolver, and that the tool-discovery expected set includes it.
7. THE former static-Dockerfile test SHALL be dropped or repurposed; the reproducibility test SHALL
   relax from version-match to digest + Resolved_Requirements.

## Out of Scope

- Per-framework AWS images (CatBoost/LightGBM built-in images) — rejected in `decision.md`; the
  open catalogue is reached via Requirements on the one Base_Image.
- A custom Docker image or any ECR push by us.
- Sandboxing installed packages beyond the job's existing IAM role (trusted-user deployment).
- Pinning any library version to the owner's current set.

## Dependencies

- Consumed by the open catalogue in the **ensembles-and-learners** spec (installed learners) and by
  **free-form-experiments** (per-experiment Requirements). Those specs assume this one has landed.
- Interacts with **cross-validation** (now complete): the CatBoost removal and the reproducibility
  wording touch tests CV added, so this lands after CV (done) to churn those files once.

## Open Questions for the Owner

| # | Question | Working assumption |
|---|---|---|
| 1 | Exact AWS XGBoost image version string for `image_uris.retrieve`? | Resolve the latest the SDK offers for the region at submission; allow an override setting if a specific version is ever needed. |
| 2 | Keep `container/Dockerfile` as a dev convenience or delete it? | Delete it and drop it from the manifest; re-add later if a local container is ever wanted. |
| 3 | Where do Requirements come from in the UI — free text or a small editor? | Out of this spec (UI is Spec A); here it is just an optional request field + saved artifact. |
| 4 | Should an empty-Requirements run still run `pip freeze` to record the image's set? | Yes, cheaply, so every run has a Resolved_Requirements record; may be skipped if it costs noticeable time. |
