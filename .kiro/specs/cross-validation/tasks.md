# Implementation Plan: Cross-Validation and Test Fold

## Overview

Python, following `design.md`. Order: contracts + layout → Fold_Planner → preparation → training
(fold loop, OOF, refit, resumability) → evaluation (aggregation, ranking, Test_Evaluation) →
orchestrator / service / config / budget → consumers → MCP / report / reasoning → end-to-end and
backward-compatibility tests. Every inherited-file change is additive and followed by a manifest
update. Commands: `.venv\Scripts\python -m pytest -q`, `.venv\Scripts\python -m ruff check .`,
`.venv\Scripts\python -m ruff format --check .`.

## Task Dependency Graph

```
1 tooling + contracts + layout
└─▶ 2 Fold_Planner (+ PBTs)
     └─▶ 3 checkpoint
          └─▶ 4 preparation
               └─▶ 5 training (fold loop, OOF, refit, resume)
                    └─▶ 6 checkpoint
                         └─▶ 7 evaluation (aggregation, ranking, Test_Evaluation)
                              └─▶ 8 orchestrator / service / config / budget
                                   └─▶ 9 checkpoint
                                        ├─▶ 10 consumers
                                        └─▶ 11 MCP / report / reasoning
                                             └─▶ 12 end-to-end + backward compatibility
                                                  └─▶ 13 final checkpoint
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
    { "wave": 10, "tasks": ["10", "11"], "depends_on": ["9"] },
    { "wave": 11, "tasks": ["12"], "depends_on": ["10", "11"] },
    { "wave": 12, "tasks": ["13"], "depends_on": ["12"] }
  ]
}
```

## Tasks

- [x] 1. Test tooling and contract foundations
  - [x] 1.1 Add Hypothesis to the dev extra and capture pre-feature golden fixtures
    - Add `hypothesis==<exact version>` to `[project.optional-dependencies].dev` in `pyproject.toml` (resolve the current release with `pip index versions hypothesis`, pin exactly), install into `.venv`
    - Before any contract change, generate golden fixtures under `tests/fixtures/pre_cv/` by running a Single_Split_Mode experiment through `InlineOrchestrator`: `experiment_config.json`, `validation.json`, `metadata.json` (one model), `comparison.json`, `experiment_summary.json`; add a helper that masks timestamps and durations for comparison
    - _Requirements: 13.1, 13.2, 13.6_
  - [x] 1.2 Create `ml_engine/contracts/cv.py`
    - `EvaluationMode`, `CvScheme`, `CrossValidationConfig`, `FoldAssignment` (with length/ordering/value validators), `FoldRowCounts`, `CvPreparationSummary`, `TestFoldSummary`, `FoldResult`, `FoldMetrics`, `OofPredictions`, `MetricAggregate`, `CvRunSummary`, `RefitIterations`, `RunCvRecord`, `CvComparisonInfo`, `TestRunResult`, `TestEvaluationReport`; constants `TEST = -1`, `TRAIN_ONLY = -2`
    - Import only `common`, `metrics`, `warnings` and `explain.PermutationEntry` (no cycle with `config.py`)
    - Register in `NEW` in `scripts/inherited_manifest.py`
    - _Requirements: 4.6, 12.1_
  - [x] 1.3 Add optional CV fields to inherited contracts
    - `SplitConfig.evaluation_mode`, `.cv`, `.test_fraction` (model validator: `cv` required iff mode is CV; `test_fraction` in (0, 0.5)); `PreparationReport.cross_validation`, `.test_fold`; `ModelMetadata.cross_validation`; `ModelComparisonEntry.cv`, `.test_metrics`; `ComparisonReport.cross_validation`, `.test_evaluation`
    - Declare each with `exclude_if` at its default; first add a unit test proving `exclude_if` works on the pinned Pydantic with `write_model` (fallback: a `model_serializer(mode="wrap")` that drops default-valued new keys)
    - Additive optional fields on new-file contracts: `PermutationImportance.fold_count/fold_baselines`, `AblationEntry.fold_drops/std_drop`, `AblationReport.fold_count`, `ThresholdAnalysis.source`, `ArchivedRun.evaluation_mode/cv_folds/test_fraction`
    - Move `ml_engine/contracts/config.py`, `preparation.py`, `model.py`, `comparison.py` from hash-tracked to `ADAPTED` with the justifications in design "What changes in inherited files"; run `.venv\Scripts\python scripts/inherited_manifest.py --write`
    - _Requirements: 1.1, 1.6, 1.8, 3.1, 12.2, 12.5_
  - [x]* 1.4 Write property test for contract backward compatibility
    - **Property 14: Backward compatibility of contracts**
    - Hypothesis builds of pre-feature-shaped contracts; also parse the golden fixtures from 1.1
    - **Validates: Requirements 1.8, 12.2, 12.3**
  - [x] 1.5 Add layout entries in `ml_engine/io/layout.py`
    - Constants and `ExperimentLayout` members from design "New layout entries" (fold assignment, development/test datasets, fold preprocessors, per-fold results and OOF slices, fold metrics, OOF manifest/payload, test evaluation)
    - Append `DEVELOPMENT_DATASET_FILE`, `TEST_DATASET_FILE` to `RUN_OUTPUT_FILES`; update the layout's `ADAPTED` justification; regenerate the manifest
    - Unit tests for every new path
    - _Requirements: 4.1, 5.5, 6.3, 10.6, 12.4_

- [x] 2. Fold_Planner (`ml_engine/splitting/folds.py`, NEW)
  - [x] 2.1 Implement scheme resolution, fingerprint and Test_Fold carve-out
    - `resolve_cv_scheme`, `dataset_fingerprint`, `FoldPlanningError(SplittingError)` with `reason` + details
    - `carve_test_fold` reusing `split_dataset` with `validation_fraction=test_fraction`; `TEST_FOLD_TOO_SMALL` below `MIN_ROWS_PER_FOLD`
    - Do not modify `strategies.py` or `ml_engine/splitting/__init__.py`; import the module directly; add to `NEW`
    - _Requirements: 2.2, 3.2, 3.3, 3.4, 3.5, 3.9_
  - [x] 2.2 Implement `plan_folds` and row helpers
    - `stratified_kfold` with rare-class fallback to `kfold` + HIGH warning; `kfold`; `group_kfold` (missing → one group, groups < K → `GROUPS_FEWER_THAN_FOLDS`); `forward_chaining` (K + 1 blocks on distinct times, unparseable → `TRAIN_ONLY`, `TEMPORAL_TIES`); `FOLD_TOO_SMALL` check
    - Single_Split_Mode with Test_Fold produces an assignment with `folds=0`
    - `fold_rows`, `development_rows`, `test_rows`
    - Unit tests for each edge case and each scheme on a hand-built frame
    - _Requirements: 2.1–2.9, 4.1, 4.6_
  - [x] 2.3 Write property test for determinism
    - **Property 1: Fold assignment is deterministic**
    - **Validates: Requirements 4.2**
  - [x] 2.4 Write property test for the partition
    - **Property 2: Fold assignment partitions the usable rows**
    - **Validates: Requirements 4.3, 3.2, 3.6, 2.2**
  - [x] 2.5 Write property test for group isolation
    - **Property 3: Group isolation**
    - **Validates: Requirements 2.3, 2.6, 3.4, 4.4**
  - [x] 2.6 Write property test for temporal ordering
    - **Property 4: Temporal ordering**
    - **Validates: Requirements 2.4, 2.5, 3.5, 4.5**
  - [x] 2.7 Write property test for stratified balance
    - **Property 5: Stratified folds preserve class proportions**
    - **Validates: Requirements 2.1, 3.3**
  - [x] 2.8 Write property test for the round trip
    - **Property 6: Fold assignment serialization round trip**
    - **Validates: Requirements 4.7, 4.6**

- [x] 3. Checkpoint - Ensure all tests pass
  - Run pytest, `ruff check` and `ruff format --check`; ensure all tests pass, ask the user if questions arise.

- [x] 4. Preparation stage
  - [x] 4.1 Implement fold-aware preprocessing (`ml_engine/preprocessing/folds.py`, NEW)
    - `fit_pipelines`, `fit_fold_pipelines` (per-fold fit on training rows only + one Development_Set fit); `fitted_on` set to `fold_{i}_train` / `development`
    - _Requirements: 5.1, 5.2, 5.3, 3.6_
  - [x] 4.2 Write property test for fit isolation
    - **Property 7: Preprocessing fit isolation**
    - **Validates: Requirements 5.1, 5.2, 5.3, 3.6**
  - [x]* 4.3 Write property test for validation independence
    - **Property 8: Validation independence of fold preprocessing**
    - **Validates: Requirements 5.4**
  - [x] 4.4 Implement `jobs/preprocessing/cv.py::prepare_with_folds` and the dispatch
    - Reuse target drop and `_derive_features`; carve Test_Fold; `plan_columns` on the Development_Set; CV → `plan_folds`, single → inherited `split_dataset` on the Development_Set
    - Write fold assignment, `development.parquet` (CV) or train/validation parquet (single), `test.parquet`, full and per-fold preprocessors, `PreparationReport` with `cross_validation` / `test_fold` and the CV `SplitSummary` mapping
    - Map `FoldPlanningError` → `JobError("SPLIT_FAILED", ..., details)` and fold fit errors → `PREPROCESSING_FAILED` with `fold_index`
    - Add the dispatch in `jobs/preprocessing/main.py` (only when CV or Test_Fold); move the file to `ADAPTED`; add `cv.py` to `NEW`; regenerate the manifest
    - Tests with `InMemoryObjectStore`: artifact set per mode, digest/row-id order, a write-recording store proves the job is the only writer of the assignment, derived features computed before assignment, Test_Fold too small → `SPLIT_FAILED`
    - _Requirements: 3.2, 3.6, 3.9, 4.1, 4.8, 5.1, 5.2, 5.5, 5.6, 12.6_

- [x] 5. Training stage
  - [x] 5.1 Implement OOF assembly and refit-iteration helpers (in `jobs/training/cv.py`, NEW)
    - `assemble_oof(slices, assignment)` with rejection of overlaps, gaps and foreign rows; OOF Parquet + `OofPredictions` manifest with payload hash
    - `refit_iterations(per_fold)` → `RefitIterations` (not applicable when no fold reports a best iteration)
    - _Requirements: 6.3, 6.7, 8.4_
  - [x] 5.2 Write property test for OOF coverage
    - **Property 9: Out-of-fold coverage**
    - **Validates: Requirements 6.7, 3.6**
  - [x] 5.3 Implement the fold loop with per-fold persistence and resume
    - Load and verify assignment + `development.parquet` (`FOLD_ASSIGNMENT_MISSING` / `_MISMATCH`)
    - Per fold: skip when `fold_{i}.json` matches `assignment_digest` + `params_digest` and its OOF slice exists; plugin from `plugin_factory` with checkpoint sub-dir `fold_{i}`; `train_model(..., compute_train_metrics=False)`; OOF slice aligned to Development_Set labels (`cv_fold_missing_class`); per-fold permutation via `validation_scorer`; write slice then `fold_{i}.json` (with timings, `TRAINING_JOB_NAME`, `resumed`)
    - Any fold exception → `JobError("CV_FOLD_FAILED", ..., {"failed_fit": "fold_i", "fold_index": "i", "model": m})`
    - _Requirements: 6.1, 6.2, 6.4, 6.5, 6.6, 9.4, 9.7_
  - [x] 5.4 Implement refit, thresholds, permutation aggregate and metadata
    - Write `fold_metrics.json` and OOF; refit with the Development_Set preprocessor and checkpoint sub-dir `refit` (`CV_REFIT_FAILED`); rewrite metadata per design C5 step 4 (`metrics` = CV mean, `train_metrics` = in-sample, `cross_validation` record)
    - Binary thresholds from OOF scores (`source="oof"`); `PermutationImportance` fold mean ± std (`fold="cv"`)
    - Save Refit_Model at `model_artifact`, write `metadata.json` last
    - Dispatch from `run_training` in CV_Mode; add optional `details` to `write_model_failure` and pass it from `InlineOrchestrator`; update the `ADAPTED` text for `jobs/training/main.py`; add `jobs/training/cv.py` to `NEW`; regenerate the manifest
    - _Requirements: 6.3, 8.1, 8.2, 8.3, 8.4, 10.1, 10.2_
  - [x] 5.5 Write unit tests for the training stage
    - Recording plugin: K + 1 fits, identical params; fold metric equals `evaluate` on the fold; failure on fold 2 and on refit records `failed_fit` and leaves another model's Run successful; all writes under `models/{m}/`; injected crash after fold 1 then rerun fits only remaining folds and gives equal `FoldMetrics`; checkpoint sub-dirs per fold; thresholds equal `threshold_analysis` on OOF; refit iterations (fake early-stopping plugin and not-applicable)
    - _Requirements: 6.1, 6.2, 6.4, 6.5, 6.6, 8.3, 8.4, 9.4, 10.1, 10.2_

- [x] 6. Checkpoint - Ensure all tests pass
  - Run pytest, `ruff check` and `ruff format --check`; ensure all tests pass, ask the user if questions arise.

- [x] 7. Evaluation stage
  - [x] 7.1 Implement `ml_engine/reporting/cv.py` (NEW)
    - `summarize_folds` (mean clamped to [min, max], sample std, per-metric fold counts, `cv_primary_metric_missing`), `instability_warnings` (std / CV rule; leave-one-out outlier folds per design D5, skipped when K = 2; unit test the design's worked example 0.80/0.81/0.79/0.98/0.80 flags only fold 4), `rank_cv` (mean → std → run id)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [x] 7.2 Write property test for aggregates
    - **Property 10: CV aggregates are correct and ordered**
    - **Validates: Requirements 7.1, 7.2**
  - [x] 7.3 Write property test for ranking
    - **Property 11: CV ranking order, independent of test results**
    - **Validates: Requirements 7.3, 7.4, 3.8**
  - [x] 7.4 Write property test for instability and outlier warnings
    - **Property 12: Instability and outlier warnings follow the thresholds**
    - **Validates: Requirements 7.5, 7.6**
  - [x] 7.5 Implement `jobs/evaluation/cv.py::extend_comparison` and the Test_Evaluation
    - Read `fold_metrics.json` per successful Run, attach `cv`, re-rank, append warnings, set `cross_validation`
    - Test_Evaluation once per successful Run with the Refit_Model (CV) or trained model (single) and `layout.preprocessor(strategy)`; write `comparison/test_evaluation.json`; copy `test_metrics`; per-Run `TEST_EVALUATION_FAILED`
    - Add the one-call hook in `jobs/evaluation/main.py` (only when CV or Test_Fold); move it to `ADAPTED`; add new modules to `NEW`; regenerate the manifest
    - Tests: Test_Evaluation uses the Refit_Model and runs once per successful Run; single split without Test_Fold skips the hook; evaluation is the only writer of comparison/summary/test evaluation
    - _Requirements: 3.7, 3.8, 7.3, 7.7, 7.8, 8.5_

- [x] 8. Orchestrator, service, configuration and budget
  - [x] 8.1 Add CV settings to `backend/config.py`
    - `cv_default_folds`, `cv_max_folds`, `cv_max_fits`, `cv_training_max_runtime_seconds`, `default_test_fraction`, `cv_instability_std`, `cv_instability_cv`, `cv_outlier_multiplier`, `cv_outlier_min_gap` with the validator from design C1; settings tests
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 7.5, 7.6_
  - [x] 8.2 Extend `TrainingRequest` and `ExperimentService._resolve_split`
    - New request fields; CV guard function (pure) with `CV_FOLDS_OUT_OF_RANGE`, `CV_FIT_LIMIT_EXCEEDED`, `CV_FOLDS_WITHOUT_CV_MODE`; scheme resolution; thresholds frozen into `split.cv`; Test_Fold default fraction; all checks before `archive_previous_run`
    - Service tests: defaults, frozen config contents, nothing archived/submitted on rejection
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 3.1_
  - [x] 8.3 Write property test for the fold-count guard
    - **Property 13: Fold-count guard**
    - **Validates: Requirements 1.4, 1.5**
  - [x] 8.4 Scale the CV training runtime in `SageMakerOrchestrator._training_plan`
    - CV runtime = `cv_training_max_runtime_seconds` or `training_max_runtime_seconds × (K + 1)`; `max_wait_seconds = max(spot_max_wait_seconds, runtime)`; `CV_RUNTIME_TOO_LONG` above 2,419,200 s
    - `FakeSageMaker` tests: one training job per Run, waves/caps unchanged, Spot on, botocore-valid request, reservation = rate × runtime × (K + 1) with `enforce_budget=True`, over-budget rejection with estimate and ceiling, On-Demand fallback re-plan keeps the CV runtime, preparation `SPLIT_FAILED` submits no training job
    - _Requirements: 3.9, 9.1, 9.2, 9.3, 9.5, 9.6, 9.7_

- [x] 9. Checkpoint - Ensure all tests pass
  - Run pytest, `ruff check` and `ruff format --check`; ensure all tests pass, ask the user if questions arise.

- [x] 10. Downstream consumers
  - [x] 10.1 Ablation on stored folds (`jobs/ablation/main.py`)
    - CV branch: baseline and each candidate per fold from `development.parquet` + assignment, preprocessing refit on fold training rows, fold means, `fold_drops`/`std_drop`/`fold_count`, candidate cap `(candidates + 1) × K ≤ 60` reported; never read `test.parquet`
    - Tests: baseline equals the Run's CV mean primary metric; single-split output unchanged
    - _Requirements: 10.3, 3.6_
  - [x] 10.2 Holdout evaluation and scoring
    - Holdout overlap check reads development (CV) or train/validation (single) plus `test.parquet` when present; confirm scoring needs no change
    - Tests: both use the Refit_Model and Development_Set preprocessor in CV_Mode
    - _Requirements: 10.4_
  - [x] 10.3 Post-training leakage and experiment comparison notes
    - Pass CV summaries to the post-training builder (mean primary for suspicious-performance rules, max fold reported); add `evaluation_mode`/`cv_folds`/`test_fraction` to `backend/services/analysis.py` rows and notes
    - _Requirements: 10.5_
  - [x] 10.4 Retrain archiving
    - `ArchivedRun` CV fields filled in `archive_previous_run`; test that assignment, fold preprocessors, fold metrics, OOF, CV-bearing comparison, test evaluation, development/test datasets and Refit_Model are archived and cleared together
    - _Requirements: 10.6_

- [x] 11. MCP, report and reasoning
  - [x] 11.1 MCP `start_training` and `get_comparison`
    - New parameters passed unchanged; response echoes resolved mode/K/scheme; audit detail; updated tool descriptions
    - In-memory FastMCP client tests
    - _Requirements: 1.7, 11.1_
  - [x] 11.2 Report_Generator CV and Test sections (`ml_engine/reporting/documents.py`)
    - Scheme, K, Test_Fold size, per-Run mean ± std and min/max per metric, CV-estimate label, separate Test evaluation section; tests on JSON/HTML output
    - _Requirements: 11.2, 11.3_
  - [x] 11.3 Reasoning context and numeric filter
    - `ml_engine/reasoning/cv_context.py` builds `cross_validation` from the comparison; `ml_engine/reasoning/numbers.py::drop_unverified_numbers`; wire both into `backend/services/reasoning.py` (`governed_context`, after `_drop_unknown_columns`); add modules to `NEW`
    - _Requirements: 11.4, 11.5_
  - [x] 11.4 Write property test for numeric grounding
    - **Property 15: LLM numbers are grounded in deterministic artifacts**
    - **Validates: Requirements 11.4, 11.5**

- [x] 12. End-to-end and backward compatibility
  - [x] 12.1 End-to-end CV experiment offline
    - `InlineOrchestrator` + `InMemoryObjectStore`: K = 3, two models, Test_Fold 0.2, through preparation, training, evaluation, holdout evaluation and scoring; assert artifacts, ranking by CV mean, separate test metrics; run the same frozen config twice and assert exactly equal mean primary metrics per Run
    - Implemented in `tests/backend/test_end_to_end_cv.py`
    - _Requirements: 13.1, 13.3, 13.4, 3.7, 8.5, 10.4_
  - [x] 12.2 Single-split regression against golden fixtures
    - Same Single_Split_Mode experiment as task 1.1; masked `comparison.json`, `validation.json`, `metadata.json`, `experiment_summary.json` byte-identical to the fixtures; consumers (thresholds, permutation, ablation, holdout, scoring, leakage, archive) on a pre-feature experiment produce unchanged outputs
    - Byte-identical check in `tests/backend/test_golden_single_split.py` (from task 1.1); consumer coverage in `tests/backend/test_consumers_single_split.py`
    - _Requirements: 7.7, 10.7, 12.3, 13.6_
  - [ ]* 12.3 Opt-in SageMaker reproducibility test
    - Marked `aws`, skipped unless `ML_FACTORY_AWS_TESTS=1`; two CV runs agree on each Run's mean primary metric within 1e-6 absolute or 0.1 % relative
    - _Requirements: 13.5_

- [x] 13. Final checkpoint - Ensure all tests pass
  - Run `.venv\Scripts\python -m pytest -q`, `.venv\Scripts\python -m ruff check .` and `.venv\Scripts\python -m ruff format --check .`; confirm `tests/test_inherited_manifest.py` passes with the regenerated manifest; ensure all tests pass, ask the user if questions arise.
  - Result: 1249 passed, 1 skipped (the opt-in AWS reproducibility test 12.3); ruff check + format clean; inherited manifest test passes (no inherited engine files changed this round).

## Notes

- Tasks marked `*` are optional. Properties 8 and 14 are extra to Req 13.2; the eight properties
  Req 13.2 requires (1, 2, 3, 4, 6, 7, 9, 10) are not optional.
- Property tests use Hypothesis with `max_examples=100`, `deadline=None`, `derandomize=True` and the
  tag `# Feature: cross-validation, Property N: <text>`.
- Every task that edits an inherited file updates `scripts/inherited_manifest.py` and regenerates
  `tests/inherited_manifest.json` in the same task.
