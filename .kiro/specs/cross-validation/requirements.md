# Requirements Document

## Introduction

Today every ML Factory 2.0 experiment makes a single train/validation split in the preparation
job (`jobs/preprocessing/main.py`, `ml_engine/splitting/strategies.py`). The best model is chosen
on that validation fold and its score on the same fold is reported as the headline number. The
headline is therefore optimistically biased by model selection and carries no estimate of
variance. Holdout evaluation (`jobs/holdout_evaluation`) only helps when the user supplies a
separate file.

This feature adds two opt-in capabilities:

1. K-fold cross-validation (CV) per experiment, with a fold scheme matching each existing split
   strategy (stratified K-fold, K-fold, group K-fold, forward-chaining temporal folds).
2. An untouched test fold carved out of the dataset before CV or the single split, never used for
   preprocessing fit, model selection or threshold choice, and scored exactly once at the end.

The feature preserves the platform invariants: learned transforms fit on training rows only (per
fold in CV); every number comes from the deterministic engine; artifacts are versioned Pydantic
contracts whose paths live only in `ml_engine/io/layout.py`; one writer per document; inherited
engine files change only additively and with a manifest update; nothing is auto-excluded; the
default test suite runs offline. Single-split experiments and artifacts written before this
feature keep working unchanged.

## Glossary

- **ML_Factory**: the ML Factory 2.0 platform as a whole (engine, jobs, backend and MCP server).
- **Experiment**: one dataset, target and business context under one S3 prefix (as defined in the
  `ml-factory-2` spec).
- **Run**: one model of an Experiment, with a stable run id.
- **Single_Split_Mode**: the existing evaluation scheme with one training fold and one validation
  fold.
- **CV_Mode**: the evaluation scheme introduced by this feature, in which the Development_Set is
  partitioned into K CV_Folds.
- **K**: the number of CV_Folds requested for an Experiment in CV_Mode.
- **Development_Set**: all usable rows of the Experiment that are not in the Test_Fold.
- **CV_Fold**: one of the K disjoint validation partitions of the Development_Set; for fold i, the
  Fold_Training_Rows are the Development_Set rows that the CV_Scheme assigns to training for fold i.
- **Fold_Training_Rows**: the rows used to fit preprocessing and the model for one CV_Fold.
- **Test_Fold**: an optional partition of usable rows carved out before CV or the single split and
  used only for the final Test_Evaluation.
- **Training_Only_Set**: Development_Set rows that are used for training in every CV_Fold and
  never validated. Non-empty only under `forward_chaining`: the earliest time block (it has no
  past to train on) and rows without a parseable time value.
- **CV_Scheme**: the fold-generation scheme derived from the split strategy: `stratified_kfold`
  (from `stratified_random`), `kfold` (from `random`), `group_kfold` (from `group`) and
  `forward_chaining` (from `temporal`).
- **Fold_Assignment**: the versioned artifact recording, for every usable row, whether the row is
  in the Test_Fold and which CV_Fold the row validates in, together with the seed, CV_Scheme, K and
  dataset fingerprint that produced the assignment.
- **Fold_Metrics**: the versioned artifact holding one Run's metrics on each CV_Fold.
- **CV_Summary**: the per-Run aggregate of Fold_Metrics: mean, standard deviation, minimum and
  maximum per metric, and the number of folds that succeeded.
- **OOF_Predictions**: out-of-fold predictions; for each Development_Set row, the prediction made
  by the fold model that did not train on that row.
- **Refit_Model**: the model of a Run fitted on the full Development_Set after CV, together with
  preprocessing fitted on the full Development_Set.
- **Test_Evaluation**: the single scoring of each Refit_Model (or, in Single_Split_Mode, each
  trained model) on the Test_Fold.
- **Instability_Flag**: a deterministic warning raised when a Run's CV results vary across folds
  beyond the configured threshold.
- **Fold_Count_Ceiling**: the configured maximum value of K and the configured maximum of
  (enabled models × (K + 1)) fold fits per Experiment.
- **Fold_Planner**: the engine component that produces the Fold_Assignment.
- **Preparation_Job**: the SageMaker Processing job in `jobs/preprocessing`.
- **Training_Job**: the SageMaker Training job(s) in `jobs/training` that fit a Run.
- **Evaluation_Job**: the SageMaker Processing job in `jobs/evaluation` that builds the comparison.
- **Orchestrator**: `SageMakerOrchestrator` in `backend/orchestration/sagemaker.py` and its test
  substitute `InlineOrchestrator`.
- **Experiment_Service**: `backend/services/experiments.py`, which validates training
  configuration and starts workflows.
- **MCP_Server**: the FastMCP server that exposes ML Factory tools.
- **Report_Generator**: the reporting component that produces the model-comparison report (PDF and
  JSON).
- **Consumer_Stage**: any downstream stage that reads training results: threshold analysis,
  permutation importance, ablation, holdout evaluation, batch scoring, reporting, retrain archiving,
  post-training leakage analysis and LLM interpretation.
- **Reproducibility_Tolerance**: 1e-6 absolute or 0.1% relative, as defined in Requirement 3.6 of
  the `ml-factory-2` spec.

## Requirements

### Requirement 1: Configure cross-validation

**User Story:** As a data scientist, I want to request K-fold cross-validation for an experiment,
so that model selection and the reported score rest on more than one validation fold.

#### Acceptance Criteria

1. THE Experiment_Service SHALL accept an evaluation mode of either Single_Split_Mode or CV_Mode as
   part of the split configuration.
2. WHEN the evaluation mode is omitted, THE Experiment_Service SHALL apply the configured default
   evaluation mode (see Open Question 1).
3. WHEN CV_Mode is requested without a value for K, THE Experiment_Service SHALL use the configured
   default K (see Open Question 2).
4. IF a requested K is lower than 2 or higher than the Fold_Count_Ceiling, THEN THE
   Experiment_Service SHALL reject the training request before any job is submitted and return an
   error naming the allowed range.
5. IF the number of enabled models multiplied by (K + 1) exceeds the Fold_Count_Ceiling, THEN THE
   Experiment_Service SHALL reject the training request before any job is submitted and return an
   error stating the requested and allowed fold fit counts.
6. THE Experiment_Service SHALL record the evaluation mode, K, CV_Scheme and random seed in the
   frozen ExperimentConfig before any job is submitted.
7. WHEN the `start_training` MCP tool is called with CV parameters, THE MCP_Server SHALL pass the
   evaluation mode and K to the Experiment_Service unchanged.
8. WHEN an ExperimentConfig written before this feature is read, THE Experiment_Service SHALL
   interpret the ExperimentConfig as Single_Split_Mode without a Test_Fold.

### Requirement 2: Map split strategies to CV schemes

**User Story:** As a data scientist, I want CV to respect the same grouping and time constraints as
my chosen split, so that CV does not reintroduce the leaks the split strategy prevents.

#### Acceptance Criteria

1. WHEN CV_Mode is used with the `stratified_random` strategy on a classification problem, THE
   Fold_Planner SHALL assign CV_Folds so that each CV_Fold's class proportions match the
   Development_Set class proportions to within one row per class.
2. WHEN CV_Mode is used with the `random` strategy, or with `stratified_random` on a regression
   problem, THE Fold_Planner SHALL assign CV_Folds by shuffled K-fold partitioning.
3. WHEN CV_Mode is used with the `group` strategy, THE Fold_Planner SHALL assign every row of one
   group value to the same CV_Fold.
4. WHEN CV_Mode is used with the `temporal` strategy, THE Fold_Planner SHALL build forward-chaining
   folds in which every validation row of fold i has a time value strictly later than every time
   value among the Fold_Training_Rows of fold i.
5. WHILE the `temporal` strategy is used in CV_Mode, THE Fold_Planner SHALL place rows without a
   parseable time value only among Fold_Training_Rows, consistent with the existing TemporalSplit
   rule.
6. WHILE the `group` strategy is used in CV_Mode, THE Fold_Planner SHALL treat rows with a missing
   group value as one group, consistent with the existing GroupSplit rule.
7. IF a classification class has fewer rows in the Development_Set than K under
   `stratified_kfold`, THEN THE Fold_Planner SHALL fall back to `kfold` and record a HIGH severity
   warning naming the class count and K.
8. IF the number of distinct group values in the Development_Set is lower than K under
   `group_kfold`, THEN THE Preparation_Job SHALL fail with a SPLIT_FAILED error stating the group
   count and K.
9. IF any CV_Fold would contain fewer than the minimum rows per fold, THEN THE Preparation_Job
   SHALL fail with a SPLIT_FAILED error stating the row count, K and the minimum.

### Requirement 3: Carve out an untouched test fold

**User Story:** As a model reviewer, I want a test fold that no selection decision has seen, so
that the final reported score is an unbiased estimate.

#### Acceptance Criteria

1. THE Experiment_Service SHALL accept an optional Test_Fold fraction greater than 0 and lower than
   0.5 in both Single_Split_Mode and CV_Mode.
2. WHEN a Test_Fold fraction is configured, THE Fold_Planner SHALL assign rows to the Test_Fold
   before assigning CV_Folds or the single validation fold.
3. WHEN a Test_Fold is built under the `stratified_random` strategy on a classification problem,
   THE Fold_Planner SHALL preserve class proportions in the Test_Fold.
4. WHEN a Test_Fold is built under the `group` strategy, THE Fold_Planner SHALL place every row of
   one group value either entirely in the Test_Fold or entirely in the Development_Set.
5. WHEN a Test_Fold is built under the `temporal` strategy, THE Fold_Planner SHALL place the latest
   rows in the Test_Fold so that every Test_Fold time value is strictly later than every
   Development_Set time value.
6. THE ML_Factory SHALL exclude Test_Fold rows from every preprocessing fit, model fit, model
   ranking, threshold choice, permutation importance computation and ablation comparison.
7. WHEN all Runs of an Experiment have finished training, THE Evaluation_Job SHALL perform the
   Test_Evaluation once per successful Run.
8. THE Evaluation_Job SHALL report Test_Evaluation metrics in a section separate from the CV or
   validation metrics and SHALL leave the model ranking unchanged by Test_Evaluation results.
9. IF the configured Test_Fold fraction yields fewer than the minimum rows per fold, THEN THE
   Preparation_Job SHALL fail with a SPLIT_FAILED error stating the resulting Test_Fold row count,
   and THE Orchestrator SHALL halt the Experiment before any Training_Job is submitted so that the
   user reconfigures the Test_Fold fraction (no silent fallback to an Experiment without a
   Test_Fold).

### Requirement 4: Produce a reproducible fold assignment

**User Story:** As an auditor, I want the exact fold of every row recorded, so that the experiment
can be reproduced and every consumer uses identical folds.

#### Acceptance Criteria

1. WHEN the Preparation_Job runs, THE Fold_Planner SHALL write one Fold_Assignment artifact per
   Experiment at a path defined in `ml_engine/io/layout.py`.
2. FOR ALL datasets and seeds, running the Fold_Planner twice with the same dataset, configuration
   and seed SHALL produce identical Fold_Assignments (determinism property).
3. FOR ALL Fold_Assignments, every usable row SHALL belong to exactly one of the Test_Fold, the
   Training_Only_Set, or one CV_Fold validation partition; the Training_Only_Set SHALL be empty for
   every CV_Scheme other than `forward_chaining`; and the union of all partitions SHALL equal the
   set of usable rows (partition property).
4. FOR ALL Fold_Assignments under `group_kfold`, no group value SHALL appear in two different
   CV_Folds or in both the Test_Fold and the Development_Set (group isolation property).
5. FOR ALL Fold_Assignments under `forward_chaining`, the maximum time value among Fold_Training_Rows
   of fold i SHALL be strictly lower than the minimum time value among validation rows of fold i
   (temporal ordering property).
6. THE Fold_Assignment SHALL record the dataset fingerprint, CV_Scheme, K, Test_Fold fraction,
   random seed and schema version.
7. FOR ALL valid Fold_Assignment objects, serialising then parsing the artifact SHALL produce an
   equal object (round-trip property).
8. THE Preparation_Job SHALL be the only writer of the Fold_Assignment.

### Requirement 5: Isolate preprocessing per fold

**User Story:** As a data scientist, I want imputers, scalers and encoders fitted only on each
fold's training rows, so that no validation information leaks into the CV estimate.

#### Acceptance Criteria

1. WHILE an Experiment is in CV_Mode, THE Preparation_Job SHALL fit each required preprocessing
   strategy separately for each CV_Fold using only that CV_Fold's Fold_Training_Rows.
2. WHILE an Experiment is in CV_Mode, THE Preparation_Job SHALL fit each required preprocessing
   strategy once more on the full Development_Set for the Refit_Model.
3. FOR ALL CV_Folds, the fitted statistics of a fold's preprocessing pipeline SHALL equal the
   statistics obtained by fitting the same pipeline on that fold's Fold_Training_Rows alone
   (fit isolation property).
4. FOR ALL CV_Folds, changing values only in that fold's validation rows SHALL leave that fold's
   fitted preprocessing pipeline unchanged (validation independence property).
5. THE Preparation_Job SHALL store each fold's fitted pipeline at a fold-indexed path defined in
   `ml_engine/io/layout.py`.
6. THE Preparation_Job SHALL apply stateless row-wise derived features before fold assignment,
   consistent with Requirement 13.3 of the `ml-factory-2` spec.

### Requirement 6: Train and score each fold

**User Story:** As a data scientist, I want every enabled model trained and scored on every fold,
so that I can see how stable each model's performance is.

#### Acceptance Criteria

1. WHILE an Experiment is in CV_Mode, THE Training_Job SHALL fit each enabled model once per CV_Fold
   on that CV_Fold's Fold_Training_Rows with the Run's hyperparameters unchanged across folds.
2. WHEN a fold model has been fitted, THE Training_Job SHALL compute that model's metrics on that
   CV_Fold's validation rows using the existing deterministic evaluation functions.
3. WHEN all CV_Folds of a Run have been scored, THE Training_Job SHALL write the Run's Fold_Metrics
   and OOF_Predictions as versioned artifacts under the Run's own prefix.
4. IF a single CV_Fold of a Run fails, THEN THE Training_Job SHALL mark that Run as failed, record
   the failing fold index and the error, and leave the other Runs unaffected.
5. WHEN THE Training_Job records a Run failure in CV_Mode, THE Training_Job SHALL identify the
   failing fit (the CV_Fold index or the Refit_Model fit) in the failure record.
6. THE Training_Job SHALL write only artifacts belonging to its own Run.
7. FOR ALL Runs in CV_Mode, the OOF_Predictions SHALL contain exactly one prediction for every
   Development_Set row that falls in a CV_Fold validation partition, and zero predictions for
   Test_Fold rows (coverage property).

### Requirement 7: Aggregate and rank by cross-validated score

**User Story:** As a decision maker, I want models ranked by mean CV score with the spread shown,
so that I pick a model that is good on average and not just on one lucky fold.

#### Acceptance Criteria

1. WHEN all Runs have finished, THE Evaluation_Job SHALL compute for each successful Run a
   CV_Summary with mean, sample standard deviation, minimum and maximum of every metric across
   CV_Folds.
2. FOR ALL Fold_Metrics inputs, the CV_Summary minimum SHALL be lower than or equal to the mean and
   the mean SHALL be lower than or equal to the maximum (aggregate ordering property).
3. WHILE an Experiment is in CV_Mode, THE Evaluation_Job SHALL rank Runs by the mean of the primary
   metric across CV_Folds, using the primary metric's optimisation direction.
4. WHEN two Runs have equal mean primary metric, THE Evaluation_Job SHALL order the Runs by lower
   standard deviation and then by run id.
5. WHEN either the coefficient of variation of a Run's primary metric exceeds the configured
   coefficient-of-variation threshold or the standard deviation exceeds the configured
   standard-deviation threshold, THE Evaluation_Job SHALL attach an Instability_Flag
   warning to that Run stating the standard deviation and the fold values.
6. WHEN K is at least 3 and one CV_Fold's primary metric for a Run deviates from the mean of that
   Run's other K − 1 folds by more than the larger of (the configured outlier multiplier × the
   sample standard deviation of those other folds) and the configured minimum outlier gap, THE
   Evaluation_Job SHALL record a warning naming the fold index
   and recommending review of that fold for leakage or distribution shift.
7. WHILE an Experiment is in Single_Split_Mode, THE Evaluation_Job SHALL produce the comparison
   artifact with the same fields and ranking as before this feature.
8. THE Evaluation_Job SHALL be the only writer of the comparison and summary artifacts.

### Requirement 8: Refit the final model

**User Story:** As a user who scores new data, I want one final model per run trained on all
development data, so that the deployed artifact uses every available row.

#### Acceptance Criteria

1. WHEN all CV_Folds of a Run have been scored successfully, THE Training_Job SHALL fit the
   Refit_Model on the full Development_Set with the Run's hyperparameters.
2. THE Training_Job SHALL store the Refit_Model at the same layout path that Single_Split_Mode uses
   for a Run's trained model.
3. THE Training_Job SHALL record in the Run's training record that the Refit_Model's reported
   performance is the CV estimate and not a measurement of the Refit_Model itself.
4. WHERE the Refit_Model is trained with early stopping, THE Training_Job SHALL set the number of
   iterations to the maximum of the per-fold best iteration counts and record the per-fold values
   and the chosen value in the Run's training record.
5. WHEN a Test_Fold exists, THE Evaluation_Job SHALL compute the Test_Evaluation metrics using the
   Refit_Model.

### Requirement 9: Execute folds within the compute and cost limits

**User Story:** As a budget owner, I want CV to stay inside the existing concurrency, Spot and
budget controls, so that a K-fold experiment cannot silently multiply cost.

#### Acceptance Criteria

1. THE Orchestrator SHALL execute fold fits using the job granularity fixed in the design (one
   Training_Job per Run with sequential folds, or one Training_Job per Run and fold; see Open
   Question 3).
2. WHILE fold fits are running, THE Orchestrator SHALL keep the number of concurrent Training_Jobs
   at or below both `max_parallel_training_jobs` and the platform-wide cap.
3. THE Orchestrator SHALL submit CV Training_Jobs with Managed Spot Training enabled by default,
   following Requirement 4 of the `ml-factory-2` spec.
4. WHERE a model supports native checkpointing, THE Training_Job SHALL resume an interrupted CV Run
   without refitting CV_Folds whose Fold_Metrics were already persisted.
5. WHEN a training request is in CV_Mode, THE Experiment_Service SHALL compute the cost estimate
   for all (K + 1) fits per enabled model and enforce the per-experiment and per-month budget
   ceilings before submission.
6. IF the CV cost estimate exceeds a budget ceiling, THEN THE Experiment_Service SHALL reject the
   training request and return the estimate and the ceiling.
7. THE Orchestrator SHALL record each fold fit's SageMaker job name, status, durations and Spot
   usage in the Run's tracking record.

### Requirement 10: Keep downstream consumers correct

**User Story:** As a platform owner, I want every stage that reads training results to behave
correctly in CV_Mode, so that thresholds, importance, ablation and scoring stay trustworthy.

#### Acceptance Criteria

1. WHILE an Experiment is in CV_Mode, THE threshold analysis SHALL compute thresholds from the
   Run's OOF_Predictions.
2. WHILE an Experiment is in CV_Mode, THE permutation importance computation SHALL use the
   Fold_Assignment and report the mean and standard deviation of importance across CV_Folds.
3. WHEN an ablation comparison runs, THE ablation job SHALL use the Experiment's stored
   Fold_Assignment so that baseline and variant are evaluated on identical folds, consistent with
   Requirement 13.5 of the `ml-factory-2` spec.
4. WHEN holdout evaluation or batch scoring runs for a CV_Mode Run, THE holdout evaluation job and
   the scoring job SHALL use the Refit_Model and the Development_Set preprocessing.
5. WHEN the post-training leakage phase runs for a CV_Mode Experiment, THE leakage service SHALL
   evaluate suspicious performance against the CV_Summary of each Run.
6. WHEN a Run is archived on retrain, THE retrain archiving step SHALL archive the Run's
   Fold_Assignment reference, Fold_Metrics, CV_Summary, OOF_Predictions and Refit_Model together.
7. WHEN any Consumer_Stage reads artifacts of an Experiment created before this feature, THE
   Consumer_Stage SHALL produce the same outputs as before this feature.

### Requirement 11: Report CV results to users and the LLM

**User Story:** As a user in chat, I want results shown as mean ± standard deviation with the test
score separate, so that I understand both performance and its uncertainty.

#### Acceptance Criteria

1. WHILE an Experiment is in CV_Mode, THE `get_comparison` MCP tool SHALL return, per Run, the mean
   and standard deviation of every metric, the per-fold primary metric values, the Instability_Flag
   status and the Test_Evaluation metrics when a Test_Fold exists.
2. WHILE an Experiment is in CV_Mode, THE Report_Generator SHALL include the CV_Scheme, K, the
   Test_Fold size, per-Run mean ± standard deviation and min/max per metric, and the Test_Evaluation
   metrics in a separate section.
3. THE Report_Generator SHALL label the Refit_Model's performance as the CV estimate.
4. WHEN the LLM interpretation is generated for a CV_Mode Experiment, THE reasoning service SHALL
   pass the CV_Summary, Instability_Flags and Test_Evaluation metrics from deterministic artifacts
   as the only numeric inputs.
5. IF the LLM interpretation references a metric value absent from the deterministic artifacts,
   THEN THE reasoning service SHALL drop that statement before storage, consistent with
   Requirement 16.2 of the `ml-factory-2` spec.

### Requirement 12: Contracts and backward compatibility

**User Story:** As a maintainer, I want CV added through versioned, additive contract changes, so
that existing experiments, artifacts and inherited-engine guarantees remain intact.

#### Acceptance Criteria

1. THE ML_Factory SHALL define the Fold_Assignment, Fold_Metrics, CV_Summary and OOF_Predictions
   artifacts as versioned Pydantic contracts.
2. THE ML_Factory SHALL add CV fields to existing contracts (SplitConfig, PreparationReport,
   comparison) only as optional fields with defaults that reproduce Single_Split_Mode.
3. FOR ALL artifacts written before this feature, parsing with the updated contracts SHALL succeed
   and produce Single_Split_Mode semantics (backward compatibility property).
4. THE ML_Factory SHALL define every new artifact path in `ml_engine/io/layout.py` only.
5. WHEN an inherited engine file changes for this feature, THE change SHALL be additive and THE
   change SHALL include an updated entry in `tests/inherited_manifest.json` with a written
   justification.
6. THE ML_Factory SHALL leave all features in the user's feature selection in place in CV_Mode and SHALL
   exclude no feature automatically on the basis of CV results.

### Requirement 13: Offline, deterministic testing

**User Story:** As a developer, I want the CV feature covered by offline deterministic tests, so
that fold correctness is verified on every change without AWS.

#### Acceptance Criteria

1. THE CV test suite SHALL run with InMemoryObjectStore, InlineOrchestrator and FakeSageMaker and
   without network access.
2. THE CV test suite SHALL include property-based tests for the determinism, partition, group
   isolation, temporal ordering, fit isolation, coverage, aggregate ordering and round-trip
   properties of Requirements 4, 5, 6 and 7.
3. THE CV test suite SHALL include an end-to-end test that runs a CV_Mode Experiment with a
   Test_Fold through preparation, training, evaluation, holdout evaluation and scoring.
4. WHEN the same frozen CV_Mode configuration runs twice in the offline test suite, THE mean primary
   metric of each Run SHALL be exactly equal across the two executions.
5. WHEN the same frozen CV_Mode configuration runs twice on SageMaker, THE mean primary metric of
   each Run SHALL match within the Reproducibility_Tolerance, consistent with Requirement 3.6 of
   the `ml-factory-2` spec.
6. THE CV test suite SHALL include a test that a Single_Split_Mode Experiment produces outputs equal
   to the outputs produced before this feature.

## Out of Scope

- Hyperparameter optimisation and nested CV.
- Repeated K-fold and Monte Carlo CV.
- Deep learning and time-series forecasting models.
- Using the Test_Fold for any selection, tuning or threshold decision.
- Statistical significance testing between models (may follow later).

## Open Questions for the Owner

All questions are resolved with the working assumptions below. They are recorded as design
decisions D1–D11 in `design.md`; the owner can override any of them.

| # | Question | Status | Resolution (working assumption) |
|---|---|---|---|
| 1 | Is CV opt-in, or on by default for new experiments? | Resolved (D1) | Opt-in; Single_Split_Mode stays the default. |
| 2 | Default K and the Fold_Count_Ceiling (max K, max models × (K + 1) fits)? | Resolved (D2) | Default K = 5; max K = 10; max 60 fits per experiment. Settings `ML_FACTORY_CV_DEFAULT_FOLDS`, `ML_FACTORY_CV_MAX_FOLDS`, `ML_FACTORY_CV_MAX_FITS`. |
| 3 | Job granularity: one Training_Job per Run with sequential folds, or one per Run × fold? | Resolved (D3) | One Training_Job per Run; folds run sequentially inside it, then the refit. Each fold's results are persisted as it finishes so a restarted job skips them. |
| 4 | Test_Fold default: off or on, and default fraction? | Resolved (D4) | Off by default; 0.2 when enabled without a fraction. |
| 5 | Should a Test_Fold be allowed in Single_Split_Mode (train / validation / test)? | Resolved (D4) | Yes. |
| 6 | Instability threshold values (either measure triggers the flag, Requirement 7.5)? | Resolved (D5) | Instability: std of the primary metric > 0.05 absolute, or coefficient of variation > 10%. Outlier fold (owner decision 2026-10-26): compared against the other K − 1 folds only, flagged when its distance from their mean exceeds max(3 × their std, 0.02); not checked when K = 2. Replaces "> 2 std from the mean", which cannot fire for K ≤ 5. All configurable. |
| 7 | Early-stopping models (XGBoost/CatBoost): iteration count for the Refit_Model? | Resolved (D6) | Maximum of per-fold best iterations; recorded as not applicable when a model does not early-stop. |
| 8 | When one fold fails, fail the whole Run or report K − 1 folds with a warning? | Resolved (D7) | Fail the Run. |
| 9 | Forward-chaining layout: expanding window only, or also sliding window? Gap between train and validation? | Resolved (D8) | Expanding window, no gap. |
| 10 | Should permutation importance in CV_Mode run on every fold or on the Refit_Model with OOF data? | Resolved (D9) | Every fold, mean ± std across folds. |
| 11 | Requirement 4.3 for `forward_chaining`: rows that can never be validated (earliest time block, rows without a parseable time)? | Resolved (D11, owner decision 2026-10-26) | They form the Training_Only_Set; Requirement 4.3 reworded so each usable row is in exactly one of Test_Fold, Training_Only_Set or one CV_Fold. No rows are dropped. |
