# Requirements Document

ML Factory 2.0 — cloud-native ML Factory on AWS SageMaker + S3, exposed through a custom MCP
server for LibreChat.

## Introduction

A new project that inherits the proven deterministic ML engine and its Pydantic contracts from
`old_repo` and runs it on AWS only. SageMaker is the sole execution backend; S3 is the sole
artifact store. There is no local execution path: no file queue, no worker, no
`LocalObjectStore`, no `file://`. Tests still run offline by mocking AWS.

The guiding rule: the inherited engine and contracts are the specification, not code to improve.
Every number comes from pandas/scikit-learn/model libraries or the tracking service; the LLM only
narrates deterministic artifacts.

## Glossary

- **Experiment**: one dataset + target + business context; a self-contained S3 prefix; holds
  one or more runs.
- **Run**: one model trained once (= one SageMaker Training Job), stable id, one experiment.
- **Stage**: profiling, preprocessing, training, evaluation.
- **Seam**: a Protocol (`ObjectStore`, `ExperimentOrchestrator`) that production code depends on
  and tests substitute.
- **Inherited engine**: `ml_engine/*` as copied from `old_repo`.

## Requirements

## Requirement 1 — Preserve the proven engine and contracts

**User story:** As the platform owner, I want the inherited engine and contracts to behave exactly
as they did, so that the numbers users trust carry over unchanged.

#### Acceptance criteria
1. THE inherited `ml_engine` packages (contracts, profiling incl. association, leakage, features,
   preprocessing, splitting, models, evaluation, reporting, reasoning, training, dictionary) SHALL
   be copied without algorithmic change.
2. THE inherited engine and job tests SHALL pass in the new project, with only fixture changes
   needed to replace the local store with a test store.
3. AWS concerns SHALL live outside `ml_engine`, except the injected-client `S3ObjectStore` in
   `ml_engine/io`.
4. AN architecture-boundary test SHALL enforce that `ml_engine` does not import `boto3` at module
   import time, the SageMaker SDK, `fastapi`, `backend`, `jobs`, or `mcp`.
5. ANY later change to an inherited engine file SHALL be deliberate: a manifest test SHALL flag
   inherited files whose content changed until the manifest is updated in a reviewed change.

## Requirement 2 — S3 as the only artifact store

**User story:** As an MLOps engineer, I want every artifact in S3, so that SageMaker jobs and the
control plane share one store.

#### Acceptance criteria
1. THE system SHALL provide `S3ObjectStore` implementing the `ObjectStore` Protocol, with the
   boto3 client injected.
2. THE production artifact store SHALL always be `S3ObjectStore`; there SHALL be no scheme-based
   store selection and no filesystem backend.
3. `ml_engine/io/uri.py` SHALL parse and validate `s3://bucket/key` URIs and reject other schemes.
4. THE artifact layout (`ml_engine/io/layout.py`) SHALL be the single source of artifact paths; an
   experiment SHALL be a self-contained S3 prefix.
5. A contract written and read back through `S3ObjectStore` SHALL round-trip byte-identically.
6. THE `ObjectStore` Protocol SHALL remain, so tests can use an in-memory store; the in-memory
   store SHALL live under `tests/` and SHALL pass the same store contract tests as
   `S3ObjectStore`.
7. S3 writes SHALL use server-side encryption (SSE-S3 or SSE-KMS, configurable).

## Requirement 3 — SageMaker as the only execution backend

**User story:** As an ML engineer, I want every stage to run as a SageMaker job, so that compute is
ephemeral and nothing runs on the control plane.

#### Acceptance criteria
1. THE system SHALL provide `SageMakerOrchestrator` implementing `ExperimentOrchestrator`; it SHALL
   be the only production implementation.
2. THE stage entrypoints (`jobs/*/main.py`) SHALL run in a SageMaker container without changes to
   their ML algorithms.
3. WHEN a workflow starts THEN the orchestrator SHALL return an `ExecutionHandle` immediately and
   SHALL NOT block on job completion.
4. THE training configuration SHALL be frozen and written to S3 before any job is submitted.
5. ONE SageMaker Training Job SHALL correspond to one model run; a failure in one SHALL NOT abort
   the others.
6. WHEN the same frozen config runs twice on SageMaker THEN the primary metric SHALL match within
   1e-6 absolute OR 0.1% relative; bitwise equality SHALL NOT be required.
7. THE default test suite SHALL run with no AWS account and no network, using a test
   orchestrator and mocked AWS clients; real-AWS tests SHALL be opt-in behind an env flag.

## Requirement 4 — Managed Spot Training by default

**User story:** As a budget owner, I want training on Managed Spot by default, so that cost is
minimized without silent escalation.

#### Acceptance criteria
1. Training jobs SHALL enable Managed Spot Training by default.
2. THE system SHALL set instance type/count, `MaxRuntimeInSeconds`, `MaxWaitTimeInSeconds`
   (default 3600, configurable), retry and concurrency limits, and a checkpoint S3 path for models
   that checkpoint.
3. XGBoost and CatBoost SHALL write native iteration checkpoints and resume from them; logistic
   regression, elastic net and random forest SHALL be documented as restart-on-interruption. THE
   system SHALL NOT claim resumability a model lacks.
4. ON Spot capacity failure THE system SHALL fail and report by default; On-Demand fallback SHALL
   be an explicit per-experiment opt-in and SHALL NEVER happen silently.
5. Instances SHALL be CPU only: default `ml.m5.large`, configurable up to a ceiling (default
   `ml.m5.2xlarge`); requests above the ceiling SHALL be rejected before submission.
6. THE system SHALL NOT create inference endpoints.
7. Per-experiment and per-month budget ceilings SHALL be configuration and SHALL be enforced
   before job submission using the cost estimate; a job whose estimate would exceed a ceiling
   SHALL NOT be submitted.

## Requirement 5 — Experiment tracking

**User story:** As a data scientist, I want durable experiment and run records, so that I can find,
compare and reproduce past work independently of chat history.

#### Acceptance criteria
1. Tracking SHALL be built on versioned contracts stored under the experiment prefix; an
   experiment SHALL be reproducible from its frozen config and artifacts.
2. THE system SHALL support listing, searching, filtering and ownership of experiments.
3. AN experiment MAY contain multiple runs; each run SHALL have a stable id and belong to exactly
   one experiment.
4. THE record SHALL capture: dataset identity/fingerprint, selected/excluded features, leakage
   findings, feature decisions, model configs, hyperparameters, splits, SageMaker job names/ARNs,
   metrics, comparisons, durations, compute config, artifact URIs, cost estimates, failures.
5. Parallel runs SHALL each write only their own per-run record; one designated aggregation step
   SHALL compose shared views. No two writers SHALL mutate the same document.
6. THE system SHALL NOT introduce a relational database without a justified need.
7. MLflow is deferred and optional; ML Factory artifacts SHALL remain authoritative.

## Requirement 6 — Training status monitoring

**User story:** As a user, I want to see where training is, so that I can tell waiting-for-Spot from
training from failed without reading raw AWS output.

#### Acceptance criteria
1. THE system SHALL use `DescribeTrainingJob`, `DescribeProcessingJob`, `ListTrainingJobs` and
   `StopTrainingJob`, and persist observations independently of chat history.
2. THE system SHALL record raw `TrainingJobStatus`, `SecondaryStatus`,
   `SecondaryStatusTransitions`, `FailureReason`, timestamps, instance and checkpoint config.
3. THE system SHALL expose a normalized status (Queued, Waiting for capacity, Preparing,
   Training, Interrupted/recovering, Evaluating, Completed, Failed, Stopped, Unknown) WHILE
   preserving the raw status.
4. WHEN AWS evidence is insufficient THEN the status SHALL be Unknown; the system SHALL NOT invent
   an intermediate status and SHALL tolerate missing or delayed fields.

## Requirement 7 — Spot utilization, savings and cost

**User story:** As a budget owner, I want measured Spot usage and clearly-labelled estimates, so
that I can report savings honestly.

#### Acceptance criteria
1. PER training job THE system SHALL record Spot enablement, instance type/count,
   `TrainingTimeInSeconds`, `BillableTimeInSeconds`, waiting time where measurable, interruption
   evidence, checkpoint usage and final status.
2. WHERE both times are present and `TrainingTimeInSeconds > 0` THE system SHALL compute
   `(1 - BillableTimeInSeconds / TrainingTimeInSeconds) * 100`, labelled a time ratio, not an
   invoice discount; otherwise the savings SHALL be Unknown.
3. Dollar estimates SHALL use a static, versioned regional price table (JSON) shipped with the
   project; every estimate SHALL record price source, table version and retrieval date, and be
   labelled an estimate, never the actual bill. Unknown instance/region SHALL yield Unknown, never
   a guessed price.
4. Training cost SHALL be separated from S3/Processing/CloudWatch costs.
5. THE system SHALL NOT claim an exact interruption count unless AWS events support it.
6. ALL cost/savings numbers SHALL come from the tracking service, never from the LLM.
7. THE system SHALL answer per-run, per-experiment and monthly usage/savings/cost queries.

## Requirement 8 — Failure diagnostics

**User story:** As a user whose training failed, I want an evidence-based explanation, so that I
can fix it without reading raw logs.

#### Acceptance criteria
1. THE system SHALL collect `FailureReason`, status transitions, stopping config, bounded
   CloudWatch logs, checkpoint metadata, the stage's `failure.json`, and exit codes where available.
2. A deterministic classifier SHALL categorize: insufficient capacity, Spot wait timeout, container
   startup, missing dependency, invalid data, OOM, IAM, S3 access, script exception, unknown.
3. THE LLM MAY explain observed evidence and suggest fixes but SHALL NEVER invent a cause.
4. Log retrieval SHALL be bounded (lines and bytes) and PII-redacted before any log reaches an LLM.

## Requirement 9 — Custom MCP server

**User story:** As a LibreChat integrator, I want ML Factory exposed as typed MCP tools, so that a
chat agent can drive it without duplicating business logic.

#### Acceptance criteria
1. THE MCP server SHALL use FastMCP (or the official Python MCP SDK) and call backend services;
   it SHALL hold no ML algorithm or business logic.
2. Tools SHALL cover dataset/experiment, analysis, feature engineering, training, and
   results/tracking operations, using the contracts verbatim.
3. State-changing tools SHALL require authorization; read-only tools SHALL be marked as such.
4. Long-running tools SHALL return an execution id immediately and never block on training.
5. Tools SHALL return metadata, summaries, bounded previews and artifact references — never a full
   dataset.
6. THE MCP SHALL NOT expose arbitrary AWS CLI or SDK execution.
7. An in-process MCP test client SHALL exercise every tool with no AWS and no corporate
   infrastructure.

## Requirement 10 — Secure file ingestion (CSV/TSV/Parquet only)

**User story:** As a user, I want to upload a dataset securely, so that it becomes an experiment
dataset.

#### Acceptance criteria
1. THE system SHALL ingest CSV/TSV/Parquet with delimiter/encoding detection, invalid-file
   rejection, size limits, duplicate-column detection, schema discovery, target selection,
   fingerprinting, and storage under the experiment prefix in S3.
2. Uploads SHALL reach S3 via a backend upload endpoint or a pre-signed URL issued by the
   backend; the MCP SHALL accept only authorized file references.
3. THE system SHALL NOT implement database discovery or SQL generation.
4. THE system SHALL NOT assume a LibreChat attachment is readable by the MCP, nor that any local
   path exists on a server.

## Requirement 11 — Experiment business context

**User story:** As a user, I want to state my prediction problem in plain language once, so that
leakage, feature engineering and interpretation all use it.

#### Acceptance criteria
1. EACH experiment SHALL have one persistent `BusinessContext` record: name, owner, objective,
   target, problem type, optional entity id, prediction scenario, scoring trigger, prediction
   horizon, optional temporal columns, dataset identity, primary metric.
2. THE system SHALL NOT ask "when is the prediction made relative to the outcome?"; it SHALL ask
   for the business scenario in plain language (e.g. "predict default within 30 days using what
   is known when the loan is issued").
3. Leakage investigation SHALL default to ON and SHALL NOT require per-column availability timing.
4. Business context SHALL be stored once and reused by feature engineering, leakage analysis and
   interpretation.

## Requirement 12 — Leakage investigation (highest priority)

**User story:** As a data scientist using features another team built, I want evidence-based leak
warnings I control, so that I catch leaks I cannot verify myself.

#### Acceptance criteria
1. ALL inherited deterministic rules, association calculations (|Pearson|, correlation ratio,
   Cramér's V) and the four-signal LLM prompt SHALL keep their tested behavior.
2. THE system SHALL run a pre-training phase (names, associations, deterministic rules,
   business-context reasoning) and a post-training phase (importance, suspicious performance,
   ablation, updated reasoning), persisted as separate artifacts; neither SHALL overwrite the other.
3. Statistical association SHALL be distinguished from evidence of temporal/causal leakage; high
   correlation alone SHALL NEVER establish leakage.
4. EVERY finding SHALL carry: stable id, feature, method, evidence, severity, confidence, LLM
   reasoning where applicable, recommended investigation, suggested correction, user decision,
   review status.
5. Nothing SHALL be auto-excluded except the target (and explicit technical/security exclusions);
   the user SHALL be able to keep / exclude / investigate / accept-risk / compare with-and-without.
6. Feature approval and feature selection SHALL remain distinct.
7. THE LLM SHALL present inferences as suspicion, with evidence and what would resolve it.

## Requirement 13 — Feature engineering

**User story:** As a user, I want the LLM to propose engineered features safely, so that I gain
signal without executing model-written code.

#### Acceptance criteria
1. THE LLM SHALL propose typed specifications from the closed vocabulary (ratio, difference, date
   difference, map categories, is-missing), never executable code; the inherited operators SHALL
   NOT change without justification and tests.
2. New operators MAY be added only if validatable and safe; the backend SHALL validate operator,
   source columns, types, target access, math constraints, missing behavior, leakage risk,
   naming, reproducibility.
3. Stateless row-wise derivations MAY precede the split only if they use neither the target nor
   post-prediction information; learned transforms SHALL fit on the training fold only.
4. THE system SHALL NOT invent nonexistent historical observations.
5. Ablation comparisons SHALL use identical validation folds and SHALL NOT use the holdout test
   set; each proposal SHALL carry an interpretable hypothesis.

## Requirement 14 — Reporting

**User story:** As a user, I want EDA and model-comparison reports, so that results are shareable
and machine-readable.

#### Acceptance criteria
1. Reports SHALL reuse the inherited contracts; libraries (ydata-profiling, SHAP, Plotly, Jinja2,
   WeasyPrint) SHALL be evaluated per need before adoption.
2. THE model-comparison report SHALL include algorithms, feature sets, hyperparameters, validation
   strategy, metrics, importance, ranking, duration, Spot usage, estimated cost, failures, LLM
   interpretation and follow-ups.
3. Reports SHALL be downloadable as PDF and JSON.
4. LLM narratives SHALL be grounded in deterministic results and SHALL NOT invent statistics.

## Requirement 15 — Security and governance

**User story:** As a corporate operator, I want authenticated, least-privilege, auditable
operation, so that the platform is safe in our environment.

#### Acceptance criteria
1. THE system SHALL enforce experiment ownership and user-scoped artifact access, and SHALL NOT
   trust raw identity headers without a verified auth boundary. (Mechanism deferred to the
   LibreChat integration phase; until then a single configured dev identity is used and flagged.)
2. LibreChat authentication SHALL NOT be assumed to establish ML Factory authorization.
3. AWS access SHALL use separate least-privilege IAM roles (control plane, SageMaker job, Step
   Functions); no hardcoded credentials; encryption in transit and at rest.
4. THE system SHALL enforce file-size limits, data retention, audit history, compute quotas and
   budget ceilings, and SHALL NOT execute model-generated code.
5. A governance setting SHALL control what reaches Bedrock; default: column names, association and
   importance stats, and sample values capped per column. Full datasets SHALL never be sent.
   Logs reaching an LLM SHALL be bounded and PII-redacted.

## Requirement 16 — LLM boundary (cross-cutting)

**User story:** As the platform owner, I want the LLM strictly bounded, so that trust in the
numbers is never compromised.

#### Acceptance criteria
1. THE LLM SHALL NOT compute metrics, invent statistics, modify state, generate executable code,
   auto-exclude features, claim a job completed without verified status, or claim leakage proven.
2. ALL LLM output SHALL be validated against versioned Pydantic schemas; references to
   nonexistent columns SHALL be dropped before storage.
3. THE deterministic engine and tracking service SHALL be the sole sources of numbers.
4. Bedrock SHALL be called through the inherited injected client; the configured region SHALL
   match the inference-profile prefix (e.g. `us.` → a `us-*` region) and a mismatch SHALL fail at
   startup.

## Out of scope

Local execution of any kind; Starburst/SQL ingestion; Colab MCP; HyperPod MCP; endpoint
deployment; deep learning, NLP, time series; automated/Bayesian HPO; SageMaker Pipelines; accessing
corporate LibreChat/MCP/DB/network from development.

> **Amended by the grid-search spec (Req 7.5):** "HPO" above means *automated/Bayesian*
> hyperparameter optimisation (that stays out of scope). **User-specified, user-approved grid
> search** over model parameters is in scope — see `.kiro/specs/grid-search/`.

## Open values owner must supply (working defaults in place)

| Item | Default |
|---|---|
| Spot `MaxWaitTimeInSeconds` | 3600 |
| Instance ceiling | `ml.m5.2xlarge` |
| Per-experiment / per-month budget | $25 / $500 (placeholders) |
| Bedrock model id | none (old repo ships empty); Bedrock disabled until set |
| Bedrock sample-value governance | 5 values per column, max 64 chars each |
| AWS region / account / bucket | config only, no default bucket |
