# Requirements Document

Control UI and HTTP API — a web front-end and the HTTP API behind it, as a peer of the MCP over
the one service layer.

## Introduction

ML Factory 2.0 today is driven only through its MCP server (for a chat agent such as LibreChat).
All business logic lives in `backend/services/*`; the MCP is a thin typed wrapper over those
services and holds no logic of its own. This feature adds a **second front-end**: a web UI the
owner drives directly, backed by a thin **HTTP API** over the same services. The MCP and the UI
are peers over one service layer; neither duplicates the other's logic, and both stay in sync
because both call the same functions.

The UI is the primary front-end (the owner's requirement); LibreChat remains supported and is
wired to the same deployment later if convenient. The HTTP API is the contract both the UI and
any other programmatic caller use.

This feature **surfaces the end-to-end experiment flow** the platform already implements and adds
the pieces that flow needs but does not yet have: a durable, listable set of **model versions**
produced across training and experimentation, and **validation of a chosen model version on a
labeled dataset uploaded to S3**, with threshold analysis computed on that dataset.

The platform invariants are unchanged: SageMaker-only execution; S3-only artifacts; every number
from the deterministic engine or the tracking service; the LLM may **propose** any choice
(including treatment, model, package and infrastructure choices) and narrate results, but it
**computes no number and applies nothing without explicit user approval** — it consults and waits,
it never decides or mutates on its own (platform LLM boundary, stock-image decision B12); nothing is
auto-excluded; artifacts are versioned Pydantic
contracts whose paths live only in `ml_engine/io/layout.py`; ownership and audit on every
mutation; the default test suite runs offline.

Out of band but adjacent (their own specs): training-time data treatment — outlier handling and
imputation (Spec B); grid search over model parameters (Spec C). This spec references the
**model-version store** they both rely on and defines it here, since the UI is its first
consumer.

## Glossary

- **Platform**: ML Factory 2.0 as a whole (engine, jobs, backend services, MCP, and the new API).
- **Service_Layer**: `backend/services/*` — the single home of all business logic.
- **API**: the new HTTP service that exposes Service_Layer operations over HTTP (the backend of
  the UI). Framework to be chosen in design (FastAPI is the likely fit; the errors module is
  already framework-free).
- **UI**: the web front-end the owner uses, a client of the API.
- **MCP**: the existing FastMCP server (`mcp_server/*`), the chat front-end, a peer of the UI.
- **Experiment**: one dataset + target + business context under one S3 prefix.
- **Run**: one model trained once within an experiment, with a stable id (existing meaning).
- **Model_Version**: a trained, saved model artifact plus its metadata, identified by a stable
  id, produced by a training run, a retrain, or (Spec C) a grid-search point. Every Model_Version
  is independently loadable and independently scorable on a Validation_Dataset.
- **Model_Version_Store**: the durable, listable collection of all Model_Versions of an
  experiment, under the experiment's S3 prefix. Nothing is overwritten; a new training run adds
  versions rather than replacing them.
- **Validation_Dataset**: a dataset the user uploads to S3 to score trained Model_Versions on. It
  **may carry the label column**.
- **Label_Column**: the target column; when present in a Validation_Dataset it is excluded from
  the model input and retained for scoring (post-prediction analysis).
- **Post_Prediction_Analysis**: the analysis computed after scoring a Model_Version on a
  Validation_Dataset: metrics (when labels are present), threshold analysis, row-overlap with the
  training data, and the existing post-training analyses surfaced for viewing.
- **Threshold_Analysis**: precision/recall/F1 (and related points) across decision thresholds for
  a binary classifier, computed on the labeled data at hand — "which threshold yields what".
- **Problem_Type**: binary classification, multiclass classification, or regression; inferred by
  EDA or set by the user.
- **Principal**: the authenticated caller (owner of an experiment). The API resolves it the same
  way the MCP does; the auth mechanism is deferred to the same point the MCP's is.

## Requirements

### Requirement 1: One service layer behind two front-ends

**User Story:** As the platform owner, I want the UI and the MCP to share one implementation, so
that they never diverge and no logic is duplicated.

#### Acceptance Criteria

1. THE API SHALL call `Service_Layer` functions for every operation and SHALL contain no ML or
   business logic of its own (the same rule the MCP follows).
2. FOR every capability this spec exposes, THE API and THE MCP SHALL invoke the same
   `Service_Layer` function, so a change in behavior is made in one place.
3. THE API SHALL map `AppError` subclasses to HTTP status codes with the stable `code` and the
   user-safe `message`, and SHALL NOT leak stack traces.
4. THE API SHALL enforce experiment ownership on every state-changing request through the same
   `Authorizer` the MCP uses, and SHALL NOT trust a raw identity header without the verified auth
   boundary (deferred identically to the MCP).
5. THE API SHALL record an audit event for every mutation through the existing `AuditLog`.
6. WHERE a capability already exists as an MCP tool, adding it to the API SHALL NOT require any
   change to `Service_Layer` behavior.

### Requirement 2: Start and manage experiments

**User Story:** As a user, I want to create and track experiments from the UI.

#### Acceptance Criteria

1. THE API SHALL expose creating an experiment from a dataset reference and a target column,
   returning the experiment record (wrapping `ExperimentService.create`).
2. THE API SHALL expose getting one experiment, listing experiments (optionally the caller's
   own), and soft-deleting an experiment (stopping its running work, as the service already does).
3. THE API SHALL expose setting and getting the experiment's business context (plain-language
   objective and prediction scenario).
4. WHEN an experiment is created, THE Platform SHALL start EDA automatically, as it does today.

### Requirement 3: Upload data to S3 and point an experiment at it

**User Story:** As a user, I want to upload a dataset to S3 from the UI, or point at one already
in an approved S3 location, and use it for an experiment.

#### Acceptance Criteria

1. THE API SHALL expose requesting a pre-signed S3 upload URL and registering the uploaded object
   (wrapping `DatasetService.create_upload_ticket` / `register`), so the dataset bytes never pass
   through the API or the LLM context.
2. THE API SHALL accept a dataset referenced by an `s3://` URI in an approved location, with the
   same allow-list and validation the service already applies (format, delimiter/encoding,
   size, duplicate columns).
3. THE API SHALL expose a bounded preview of a dataset's columns (capped rows and columns), never
   the full dataset.
4. THE UI SHALL let the user upload a file or paste an approved `s3://` URI, and THE API SHALL
   reject unapproved locations and malformed files with a structured error.
5. WHERE LibreChat is wired later, it SHALL use the same upload/register services; this spec does
   not block on LibreChat.

### Requirement 4: EDA and problem-type identification

**User Story:** As a user, I want the platform to profile my data and tell me the problem type,
so that I understand the dataset before engineering or training.

#### Acceptance Criteria

1. THE API SHALL expose the EDA report and the deterministic leakage findings for an experiment.
2. THE EDA SHALL infer the Problem_Type (binary / multiclass / regression); THE API SHALL expose
   the inferred type, and THE user SHALL be able to override it before training.
3. THE API SHALL expose the feature-review table (per-column stats, leakage risk, recommended
   action), reusing the existing service.
4. WHEN EDA has not finished, THE API SHALL return a structured "artifact not ready" error rather
   than fabricate a result.

### Requirement 5: Identify and approve engineered features

**User Story:** As a user, I want the platform to suggest engineered features from the EDA, and I
approve which ones are built, so that I gain signal without the model writing code.

#### Acceptance Criteria

1. THE API SHALL expose generating feature proposals (LLM-suggested typed specifications from the
   closed vocabulary) and defining features by hand, both reusing `FeatureProposalService`.
2. THE LLM SHALL only propose; THE API SHALL require explicit user approval before any feature is
   built, and nothing SHALL be auto-approved.
3. WHEN proposals are approved, THE Platform SHALL profile and leakage-screen them in a SageMaker
   job, and THE API SHALL expose their status and resulting statistics in the feature review.
4. Feature proposals SHALL require Bedrock to be enabled; WHEN it is not, THE API SHALL return a
   structured "feature disabled" error and the rest of the flow SHALL still work.
5. THE API SHALL expose the leakage investigation (pre- and post-training phases) and recording a
   user decision on each warning, reusing the existing service; nothing SHALL be auto-excluded.

### Requirement 6: Train with the user's choice of algorithm

**User Story:** As a user, I want to pick which algorithms to train and start training from the
UI, so that I control the model set.

#### Acceptance Criteria

1. THE API SHALL expose starting training with a chosen set of models, primary metric, problem
   type, class weighting, split strategy and per-model parameters (wrapping
   `ExperimentService.start_training`), returning an execution id immediately.
2. THE API SHALL expose a pre-flight check (blockers and warnings) and SHALL surface it to the UI
   before training starts.
3. THE API SHALL expose per-job execution status and the normalized training status, and failure
   diagnoses where present, reusing the tracking services.
4. THE API SHALL expose stopping a running training workflow.
5. ALL guards (instance allow-list, Spot, budget, concurrency, fold-fit ceiling) SHALL run in the
   Service_Layer exactly as they do for the MCP; the API SHALL add none and bypass none.

### Requirement 7: Persist every model version

**User Story:** As a user, I want every model produced during an experiment — each algorithm,
each retrain, each grid-search point — saved and listable, so that I can validate any of them
later.

#### Acceptance Criteria

1. THE Platform SHALL store each trained model as a Model_Version in the Model_Version_Store under
   the experiment's S3 prefix, at a path defined only in `ml_engine/io/layout.py`, with a stable
   id and its metadata (algorithm, parameters, metrics, training context, source run, created-at).
2. A new training run SHALL add Model_Versions and SHALL NOT overwrite or delete an earlier
   version's artifacts; the existing retrain-archive behavior SHALL be expressed as adding
   versions, so no prior model is lost.
3. THE Model_Version contract SHALL be a versioned Pydantic contract; THE Model_Version_Store
   SHALL have exactly one writer per version document.
4. THE API SHALL expose listing an experiment's Model_Versions with their metadata and headline
   metrics, newest first.
5. EACH Model_Version SHALL be independently loadable (its model artifact and the preprocessing
   it needs are resolvable from the version record alone).
6. WHEN an experiment created before this feature is read, THE Platform SHALL present its existing
   trained models as Model_Versions without re-training (backward compatibility).

### Requirement 8: Validate a model version on a labeled S3 dataset

**User Story:** As a user, I want to score a chosen model version — or all of them — on a dataset
I upload, which usually has labels, so that I can measure performance on data of my choosing.

#### Acceptance Criteria

1. THE API SHALL expose starting a validation of one Model_Version, or of all Model_Versions of an
   experiment, on a Validation_Dataset referenced by an approved `s3://` URI, returning at once
   with an id.
2. WHEN the Validation_Dataset contains the Label_Column, THE Platform SHALL exclude it from the
   model input before prediction, and SHALL retain it for Post_Prediction_Analysis.
3. THE Platform SHALL NOT re-fit any model or any learned transform on the Validation_Dataset; it
   SHALL apply the Model_Version and its fitted preprocessing unchanged.
4. THE Platform SHALL compute, per validated Model_Version: predictions, and — when labels are
   present — the metrics for the Problem_Type, using the existing deterministic evaluation.
5. THE Platform SHALL report the overlap between the Validation_Dataset rows and the model's
   training data, and SHALL distinguish metrics on all rows from metrics on rows not seen in
   training, as the holdout evaluation already does.
6. WHEN labels are absent, THE Platform SHALL still produce predictions and their distribution,
   and SHALL state that no metrics could be computed.
7. THE API SHALL expose the validation result per Model_Version, and a comparison across
   Model_Versions when several were validated on the same Validation_Dataset.
8. A validation SHALL run as a SageMaker job and SHALL NOT change the experiment's training state.

### Requirement 9: Post-prediction analysis, including threshold analysis

**User Story:** As a user, I want to analyze a model version's predictions on my validation data,
including which decision threshold yields which precision/recall, so that I can choose an
operating point.

#### Acceptance Criteria

1. THE API SHALL expose Post_Prediction_Analysis for a validated Model_Version: the metrics, the
   row-overlap summary, and the threshold analysis.
2. WHEN the Problem_Type is binary classification and labels are present, THE Platform SHALL
   compute Threshold_Analysis on the Validation_Dataset: precision, recall and F1 across a grid of
   thresholds, the default-threshold point, and the F1-maximizing point.
3. THE Threshold_Analysis SHALL be computed on the Validation_Dataset's own labels, and SHALL NOT
   reuse a threshold analysis computed during training.
4. THE API SHALL expose the existing post-training analyses for viewing: model comparison,
   permutation importance, ablation, the post-training leakage phase, and the LLM interpretation.
5. ALL numbers in Post_Prediction_Analysis SHALL come from the deterministic engine; the LLM MAY
   narrate them but SHALL NOT compute or alter them.
6. THE API SHALL expose downloading a validation's predictions and reports via a short-lived
   pre-signed URL scoped to the experiment, reusing the existing download service.

### Requirement 10: Reporting and downloads

**User Story:** As a user, I want shareable reports and the underlying files from the UI.

#### Acceptance Criteria

1. THE API SHALL expose generating and fetching the EDA and model-comparison reports (JSON/HTML,
   PDF when available), reusing `ReportingService`.
2. THE API SHALL expose short-lived pre-signed download links for an experiment's artifacts
   (predictions, reports, requirements.txt), scoped to that experiment only.

### Requirement 11: Security, ownership and audit

**User Story:** As a corporate operator, I want the API to be as safe as the MCP.

#### Acceptance Criteria

1. THE API SHALL enforce experiment ownership on every mutation via the existing `Authorizer`.
2. THE API SHALL NOT expose arbitrary AWS or S3 operations; it SHALL expose only the
   Service_Layer capabilities this spec names.
3. THE API SHALL record an append-only audit event for every mutation.
4. THE API SHALL apply the existing file-size, retention and governance settings; it SHALL NOT
   introduce a new path that bypasses them.
5. THE API SHALL support the corporate HTTPS proxy configuration already in `Settings`.

### Requirement 12: The UI

**User Story:** As a user, I want a web interface that walks me through the whole flow.

#### Acceptance Criteria

1. THE UI SHALL let the user: create an experiment; upload or point at an S3 dataset; view EDA,
   the inferred Problem_Type and leakage findings; view, generate and approve feature proposals;
   choose algorithms and start training; watch training status; view post-training analysis;
   list Model_Versions; start a validation on an uploaded labeled dataset; and view
   Post_Prediction_Analysis including threshold analysis.
2. THE UI SHALL call only the API; it SHALL contain no business logic and SHALL compute no metric.
3. THE UI SHALL present LLM output as suggestion/ narrative, visibly distinct from deterministic
   results, consistent with the platform's boundary.
4. THE UI SHALL never display or request a full dataset; previews are bounded.
5. WHERE a capability needs Bedrock and it is disabled, THE UI SHALL degrade gracefully and the
   rest of the flow SHALL remain usable.

### Requirement 13: Offline, deterministic testing

**User Story:** As a developer, I want the API covered offline.

#### Acceptance Criteria

1. THE API test suite SHALL run with `InMemoryObjectStore`, an in-process orchestrator and mocked
   AWS, with no network, like the MCP suite.
2. THE API tests SHALL exercise every endpoint, the error mapping, the ownership checks, the
   bounded outputs, and a full create → EDA → approve → train → validate → analyze flow.
3. Model_Version persistence and labeled-validation (label dropped for prediction, kept for
   metrics, threshold analysis on validation labels) SHALL have explicit tests.

## Out of Scope

- Outlier handling and imputation choices during training (Spec B).
- Grid search over model parameters (Spec C).
- Authentication mechanism (deferred with the MCP's; a flagged dev identity until then).
- Real-time collaborative editing, multi-tenant org management, dataset catalogs beyond the
  existing upload inbox and approved prefixes.
- Deploying models to inference endpoints (the platform creates no endpoints).

## Open Questions for the Owner

| # | Question | Working assumption |
|---|---|---|
| 1 | API framework and where it runs. | FastAPI in the same control-plane process/image as the MCP (one Fargate service), sharing the container; the MCP already has an HTTP app. |
| 2 | UI stack and hosting. | A single-page app served by the API or a static host; framework TBD in design; kept thin. |
| 3 | Validation default target: all Model_Versions or one chosen? | All, with per-version results and a comparison (owner said "validation on all of them"); a single-version call is also offered. |
| 4 | Does validating "all versions" run one SageMaker job that loops the versions, or one job per version? | Decide in design; one job looping versions keeps job count and cost down and mirrors holdout evaluation. |
| 5 | Auth for the UI (vs the MCP's deferred mechanism). | Same deferred boundary; flagged dev identity until the LibreChat/UI auth is chosen. |
| 6 | Is the existing holdout-evaluation flow folded into Model_Version validation, or kept separate? | Fold holdout evaluation into Requirement 8 so there is one "score on an uploaded dataset" path, targeting Model_Versions. |
