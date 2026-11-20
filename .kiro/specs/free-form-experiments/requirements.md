# Requirements Document

Free-form experiments — a second tier next to the managed tabular pipeline, where the user runs an
experiment the managed pipeline has no code for (a newly released classical-ML or time-series
method, a custom feature step, a new model family). The AI drafts the experiment script, the user
edits it or tells the AI what to change, approves it, and the platform runs it as a SageMaker job
with full infrastructure, tracking, reproducibility and audit around it.

## Introduction

The managed pipeline gives strong guarantees (leak-free evaluation, fold-safe preprocessing, every
number from the deterministic engine) because the platform controls every step. That makes it a
closed menu: an experiment the menu lacks cannot be run. The owner wants the platform to run **any
experiment**, for example testing a newly released method the day it appears. Scope is classical ML
and time series, not LLMs.

This spec adds a **Free_Form_Tier**. The platform does not control what the script does, so it does
not claim the managed guarantees. It provides everything *around* the script: compute (any
instance type, Spot where it works), the experiment's `requirements` install, data access, output
capture, saved code and `pip freeze`, cost and audit, and side-by-side comparison. A free-form run
is always labelled as such so its numbers are never mistaken for managed, leak-checked results.

The AI participates under the platform LLM boundary (stock-image decision B12): it **drafts** the
script, the user **edits or instructs changes**, and nothing runs until the user **approves the
exact code**. The AI computes no number and runs nothing itself.

Two bridges keep free-form results trustworthy and comparable without removing the freedom:
1. A **Platform_SDK** the script may use for the platform's own splitters, preprocessing and
   metrics, so a script that opts in gets engine-computed, fold-safe numbers.
2. **Independent validation**: any free-form run that exposes a predict entry point produces a
   Model_Version that the platform itself scores on a labelled S3 dataset, so its validation
   numbers are engine-computed and comparable with managed models.

All platform invariants hold where the platform controls them: SageMaker-only execution; S3-only
artifacts; versioned Pydantic contracts whose paths live only in `ml_engine/io/layout.py`;
ownership and audit on every mutation; the default suite runs offline; inherited engine files
change only additively with a manifest update.

## Glossary

- **Platform**: ML Factory 2.0 as a whole.
- **Managed_Tier**: the existing pipeline with engine-computed, leak-checked results.
- **Free_Form_Tier**: this feature; user-approved script, platform-run, results labelled
  self-reported unless produced through the Platform_SDK or independent validation.
- **Experiment_Script**: the code a Free_Form_Run executes — a Python entry point plus optional
  helper files.
- **Script_Draft**: an Experiment_Script proposed by the AI, not yet approved.
- **Approved_Script**: the exact code (content hash) the user approved; the only code a run may
  execute.
- **Free_Form_Run**: one execution of an Approved_Script as a SageMaker job.
- **Platform_SDK**: a small importable module available to scripts exposing the platform's
  splitters (including group and temporal splits), per-fold preprocessing, metrics and a helper to
  write results in the platform's format.
- **Self_Reported_Metric**: a number the script computed itself; stored but labelled as such.
- **Engine_Metric**: a number computed by the platform's deterministic engine, either through the
  Platform_SDK or by independent validation of the run's Model_Version.
- **Predict_Entry_Point**: a function in the script's output (a documented signature) that maps a
  feature table to predictions, enabling independent validation.
- **Model_Version**: a trained, saved model artifact plus metadata with a stable id (control-UI
  spec).
- **Requirements**: the experiment's `requirements.txt` (stock-image decision B10/B11).

## Requirements

### Requirement 1: Run a user-approved script as a platform job

**User Story:** As a data scientist, I want to run my own experiment code on the platform, so that
I can test a method the managed pipeline does not offer.

#### Acceptance Criteria

1. THE Platform SHALL accept a Free_Form experiment definition: an Approved_Script, a dataset
   reference (an approved `s3://` location), the experiment's Requirements, an instance type and
   an optional Spot flag.
2. THE Platform SHALL execute an Approved_Script as a SageMaker job on the stock base image with the
   Requirements installed first (B10), under the existing concurrency, time and budget guards.
3. THE Platform SHALL provide the script its input data location and an output location through a
   documented, versioned interface, and SHALL NOT pass dataset bytes through the API or the LLM
   context.
4. THE Platform SHALL allow any supported SageMaker instance type for the run (CPU or GPU), subject
   to the budget guard, because the experiment may need it.
5. WHEN Spot is requested, THE Platform SHALL use Managed Spot and treat an interruption as
   restart-from-scratch (no checkpoint resume; B4), and SHALL record the interruption.
6. A failed install or a failing script SHALL fail the run with a structured error and captured
   logs; it SHALL NOT silently fall back to another package set or another script.
7. THE Platform SHALL isolate runs from each other: one run SHALL NOT read or overwrite another
   run's outputs, and a run SHALL write only under its own output prefix.

### Requirement 2: The AI drafts, the user edits, the user approves

**User Story:** As a user, I want the AI to write the script and let me change it or tell it what
to change, so that I stay in control of the code that runs.

#### Acceptance Criteria

1. THE Platform SHALL let the user describe the experiment in plain language and SHALL return a
   Script_Draft proposed by the AI, together with the Requirements it proposes to install.
2. THE user SHALL be able to edit the Script_Draft directly, and SHALL be able to instruct the AI
   to revise it ("use a rolling window of 14", "add early stopping"), producing a new Script_Draft
   each time; every revision SHALL be retained.
3. THE Platform SHALL show the user the full script and Requirements before approval, with a diff
   between revisions.
4. THE Platform SHALL execute only an Approved_Script; approval SHALL bind to the content hash of
   the exact code, so any later edit requires a new approval.
5. THE AI SHALL NOT execute, approve or submit a script, and SHALL NOT approve its own revisions;
   only an explicit user action creates an Approved_Script (B12).
6. THE audit trail SHALL record each Script_Draft revision, who edited or requested it, and the
   approval (user, time, content hash).
7. THE Platform SHALL show the user the AI's stated assumptions and any risks it noticed (for
   example, "this computes a scaler on all rows"), as advisory text; absence of a warning SHALL NOT
   be presented as a guarantee of correctness.

### Requirement 3: Label results honestly: self-reported versus engine-computed

**User Story:** As a decision maker, I want to know which numbers I can trust, so that a flattering
free-form result is never mistaken for a leak-checked one.

#### Acceptance Criteria

1. THE Platform SHALL label every Free_Form_Run result as `free_form` and every metric as
   Self_Reported_Metric or Engine_Metric, and SHALL show that label wherever the metric appears.
2. THE Platform SHALL NOT present Self_Reported_Metric values in a ranking beside Engine_Metric
   values without the label, and SHALL NOT mix them in a single ranked column by default.
3. THE Platform SHALL NOT claim leak-freedom, fold-safety or leakage screening for a Free_Form_Run
   unless every number was produced through the Platform_SDK or by independent validation.
4. THE LLM MAY comment on a Free_Form_Run's results but SHALL attribute numbers to their label and
   SHALL NOT compute or alter them.

### Requirement 4: Platform SDK for trustworthy numbers (opt-in)

**User Story:** As a script author, I want to reuse the platform's splitters and metrics, so that my
experiment's numbers are comparable and leak-safe without rewriting them.

#### Acceptance Criteria

1. THE Platform SHALL provide a Platform_SDK importable by scripts, exposing the platform's
   splitters (random, stratified, group, temporal), per-fold preprocessing, the metric functions
   and a results-writer in the platform's format.
2. WHEN a script computes its reported metrics only through the Platform_SDK, THE Platform SHALL
   label them Engine_Metric.
3. THE Platform_SDK SHALL be the same code the managed pipeline uses (no parallel implementation),
   so an SDK-produced metric equals the managed pipeline's for the same inputs.
4. THE Platform_SDK SHALL import only libraries available in the base image or the run's
   Requirements, and SHALL be versioned so a run records which SDK version it used.
5. THE AI-drafted scripts SHOULD use the Platform_SDK by default for splitting and metrics, so
   drafts are comparable unless the user asks otherwise.

### Requirement 5: Time-series support

**User Story:** As a data scientist, I want to test time-series methods, so that forecasting and
temporal experiments are first-class.

#### Acceptance Criteria

1. THE Platform_SDK SHALL expose time-ordered evaluation: a temporal holdout and an
   expanding-window (rolling-origin) split, with training rows always earlier than evaluation rows
   and an optional gap.
2. THE Platform_SDK SHALL expose forecasting metrics (for example MAE, RMSE, MAPE/sMAPE and a
   seasonal-naive baseline comparison) computed by the deterministic engine.
3. THE Platform SHALL support a series identifier and a time column in the dataset definition for
   free-form runs, so multi-series data can be split without mixing future into past.
4. THE Platform SHALL refuse, with a structured error, an SDK evaluation configuration that would
   put later observations in training than in evaluation.
5. THE Platform SHALL surface a naive baseline alongside any time-series result so a model that
   fails to beat it is visible.

### Requirement 6: Save every run as an independent, reproducible model version

**User Story:** As a user, I want every free-form run kept with enough information to reproduce and
validate it later.

#### Acceptance Criteria

1. THE Platform SHALL save, per Free_Form_Run: the Approved_Script (and its content hash), the
   Requirements as installed (`pip freeze`, B11), the base image digest, the Platform_SDK version,
   the instance type and Spot setting, the run's logs, cost and timing, and the results, at paths
   defined only in `ml_engine/io/layout.py`.
2. THE Platform SHALL reproduce a run from its saved script, saved `pip freeze` and recorded image
   digest, subject to any external resources (for example downloaded weights or data) the script
   reaches; THE Platform SHALL record any external URLs the script declares.
3. WHERE the script produces a model with a Predict_Entry_Point, THE Platform SHALL register it as
   a Model_Version that is independently loadable and validatable.
4. NO run SHALL overwrite another run's saved artifacts.

### Requirement 7: Independent validation of free-form models

**User Story:** As a user, I want the platform to score a free-form model on a labelled dataset
itself, so that I can compare it fairly with managed models.

#### Acceptance Criteria

1. WHEN a Free_Form_Run exposes a Predict_Entry_Point, THE Platform SHALL support validating its
   Model_Version on an uploaded labelled dataset using the same validation flow as managed models
   (labels excluded from prediction, kept for analysis, including threshold analysis).
2. THE validation metrics SHALL be computed by the deterministic engine and SHALL be labelled
   Engine_Metric.
3. THE Platform SHALL document the Predict_Entry_Point signature and reject a registration whose
   entry point does not load or does not return predictions of the expected shape.
4. IF a Free_Form_Run exposes no Predict_Entry_Point, THEN THE Platform SHALL keep its results as
   Self_Reported_Metric only and SHALL say validation is unavailable for that run.

### Requirement 8: Compare free-form and managed runs

**User Story:** As a decision maker, I want to compare a new method with my managed models, so that
I can decide whether it is worth adopting.

#### Acceptance Criteria

1. THE Platform SHALL list Free_Form_Runs and managed runs of an experiment together, with the
   label (free_form or managed) and the metric label on every entry.
2. THE Platform SHALL offer a like-for-like comparison on Engine_Metric values only (validation
   results, or SDK-produced metrics), and SHALL state when two entries are not comparable.
3. THE Platform SHALL NOT auto-select or auto-promote any run; choosing one for downstream use is
   an explicit user action.

### Requirement 9: Adopting a free-form method into the managed tier (optional path)

**User Story:** As a maintainer, I want a path to turn a successful free-form method into a
supported managed option, so that good ideas become leak-checked and repeatable.

#### Acceptance Criteria

1. THE Platform SHALL allow the user to mark a Free_Form_Run as a candidate for adoption and to
   export its Approved_Script and Requirements as a starting point for a managed plugin.
2. ADOPTION SHALL be a normal code change (a model plugin or feature operation reviewed and tested
   like any other, per the plugin contract); THE Platform SHALL NOT convert a free-form script into
   a managed option automatically.

### Requirement 10: Guards, ownership and audit

**User Story:** As a budget owner, I want free-form runs bounded and attributable.

#### Acceptance Criteria

1. THE Platform SHALL enforce the per-experiment and monthly budget ceilings, the concurrency caps
   and a per-run maximum runtime for Free_Form_Runs before and during submission, refusing with the
   estimate and the ceiling when exceeded.
2. THE Platform SHALL enforce ownership: only the experiment's owner may create, approve, run, stop
   or read a Free_Form_Run.
3. THE audit trail SHALL record every Free_Form_Run submission, stop and outcome, with the
   approving user and the content hash.
4. THE Platform SHALL provide a stop action that terminates a running Free_Form_Run.

### Requirement 11: Contracts, paths and backward compatibility

**User Story:** As a maintainer, I want free-form support added through versioned, additive
contracts.

#### Acceptance Criteria

1. THE Platform SHALL define the Free_Form experiment definition, Script_Draft, Approved_Script
   record, run result and Predict_Entry_Point contracts as versioned Pydantic contracts.
2. THE Platform SHALL add free-form fields only as optional additions; absence reproduces today's
   managed behaviour exactly.
3. THE Platform SHALL define every new artifact path only in `ml_engine/io/layout.py`.
4. WHERE an inherited engine file changes, THE change SHALL be additive and recorded in
   `tests/inherited_manifest.json` with a written justification.

### Requirement 12: Offline, deterministic testing

**User Story:** As a developer, I want free-form support covered offline.

#### Acceptance Criteria

1. THE test suite SHALL run with `InMemoryObjectStore`, an in-process orchestrator and mocked AWS,
   with no network.
2. THE suite SHALL include tests that an unapproved or modified script is never executed, that
   approval binds to the content hash, and that the AI path cannot approve or submit.
3. THE suite SHALL include tests that a free-form result is labelled `free_form` with
   Self_Reported_Metric unless produced through the Platform_SDK or independent validation.
4. THE suite SHALL include a property-based test that every Platform_SDK temporal split places all
   training rows strictly before all evaluation rows, and a test that a violating configuration is
   refused.
5. THE suite SHALL include a test that an SDK metric equals the managed pipeline's metric on the
   same inputs, and that a Predict_Entry_Point model validates through the standard flow.
6. THE suite SHALL include tests that runs are isolated (no cross-run overwrite) and that a failed
   install fails the run without fallback.
7. THE suite SHALL include a test that a free-form run's dataset is leakage-screened before the run
   is submitted, that findings are attached to the run and shown before approval, and that a
   finding never blocks or edits the run on its own.
8. THE suite SHALL include a test that a Script_Draft cannot be produced until the data-context
   questions are answered or explicitly skipped, and that the answers and skips are stored with the
   run and passed to the drafting step.

### Requirement 13: Leakage screen on free-form inputs

**User Story:** As a data scientist, I want the platform's existing leakage screen applied to the
data a free-form run uses, so that proxy features and duplicates are caught even when the code is
mine or the AI's.

#### Acceptance Criteria

1. BEFORE a Free_Form_Run is approved, THE Platform SHALL run the existing deterministic leakage
   analysis (`ml_engine/leakage/analyzer.py::analyze_leakage`, producing a `LeakageReport`) on the
   run's dataset, target and declared feature columns, in a SageMaker job, reusing the managed
   tier's rules and `LeakageConfig` unchanged.
2. THE Platform SHALL attach the resulting findings to the run and SHALL show them to the user
   alongside the Script_Draft before approval.
3. THE leakage screen SHALL be advisory: a finding SHALL NOT block, edit, or remove a feature from
   the run on its own. The user decides whether to change the features or the script (nothing is
   auto-excluded).
4. WHEN the user approves a run with open high-severity findings, THE Platform SHALL record that
   the user acknowledged them, and SHALL show the open findings beside the run's results.
5. THE AI MAY explain the findings and propose a script change in response, under the usual
   propose-and-approve rule (B12); the numbers in the findings SHALL come only from the
   deterministic analysis.
6. IF the script declares no target or feature columns (for example an unsupervised experiment),
   THEN THE Platform SHALL run the checks that need no target (duplicate rows and entities,
   identifier-like columns) and SHALL say which checks were skipped and why.
7. WHERE the user also provides labels for analysis only (for example overlaying labels on
   clusters), THE Platform SHALL screen the features against those labels, because a proxy column
   inflates label agreement in the same way it inflates a supervised score.

### Requirement 14: AI asks about the data before drafting

**User Story:** As a user, I want the AI to ask me how the data was collected before it writes the
script, so that leaks that depend on the meaning of the data are found by my answers, not missed.

#### Acceptance Criteria

1. BEFORE producing the first Script_Draft for a run, THE AI SHALL ask the user a short set of
   data-context questions, at minimum:
   - when and how the label was assigned;
   - whether one entity (for example a client) can appear in more than one row;
   - whether any column was recorded at or after the time the outcome became known;
   - whether rows are time-ordered and which column carries the time;
   - whether any column was used to define the label.
2. THE AI MAY add questions specific to the column names and the profile summary it sees; it SHALL
   NOT see raw rows (platform invariant).
3. THE user SHALL be able to answer, answer "don't know", or skip any question; THE Platform SHALL
   record each answer, "don't know" and skip with the run.
4. THE AI SHALL use the answers when drafting: for example, a group split when entities repeat, a
   time-ordered split when rows are time-ordered, and excluding or flagging columns the user said
   were recorded after the outcome or used to define the label. THE AI SHALL state in the draft's
   assumptions which answer drove each such choice.
5. WHEN an answer is "don't know" or skipped for a question that affects the split or the feature
   set, THE AI SHALL say which risk remains open in the draft's assumptions.
6. THE saved answers SHALL be reused as defaults for later runs on the same dataset, and the user
   MAY edit them.
7. THE questions and answers SHALL be visible in the run record and the audit trail, so a reviewer
   can see what the user stated about the data.

## Out of Scope

- LLM and foundation-model experiments (GPU-heavy language models); the infrastructure allows GPU
  instances, but no LLM-specific tooling (weights management, serving) is specified here.
- Guaranteeing that arbitrary scripts are leak-free; the platform labels results instead.
- Sandboxing a script from the job's IAM role and data access beyond the existing role (the
  deployment is for the owner and trusted users).
- Automatically converting a free-form script into a managed plugin.
- Interactive notebooks / a hosted Jupyter environment; the unit is a submitted, approved script.

## Dependencies

- The **stock-image migration** and its `requirements` install layer (decision B10–B13): free-form
  runs rely on the base image plus the experiment's Requirements.
- The **model-version store and validation flow** (control-UI spec) for Requirements 6 and 7.
- The **cross-validation** spec's splitters, and the existing group and temporal splits, back the
  Platform_SDK.

## Open Questions for the Owner

| # | Question | Working assumption |
|---|---|---|
| 1 | Is "time series" forecasting of a series (predict future values) or tabular prediction with a time column, or both? | Both: the SDK offers temporal and rolling-origin splits plus forecasting metrics; deeper forecasting-specific tooling is added later as needed. |
| 2 | Should one script be able to run many configurations (a sweep) in one run? | Out for now; one approved script is one run. Sweeps are done by submitting several runs. |
| 3 | Maximum run time and cost per free-form run? | Use the existing budget guards plus a configurable per-run runtime cap. |
| 4 | Should an AI-drafted script default to the Platform_SDK? | Yes (Requirement 4.5), unless the user asks for something the SDK cannot do. |
| 5 | Can a script reach the internet to download data or pretrained weights (through the corporate proxy)? | Yes, via the existing proxy support; any declared external URLs are recorded in the run. |
