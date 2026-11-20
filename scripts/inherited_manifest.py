"""Record or verify the hashes of code inherited verbatim from old_repo.

The inherited engine is the specification (requirements 1.1, 1.5). A change to one of these files
must be deliberate: edit the file, explain why in the change, then re-run with ``--write``.

    python scripts/inherited_manifest.py --write     # re-record after a reviewed change
    python scripts/inherited_manifest.py --verify-against ../old_repo  # prove it is verbatim

Hashes are taken over the text with line endings normalised to ``\\n`` so a Git checkout on
Windows (CRLF) and Linux (LF) hash the same.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "inherited_manifest.json"
PACKAGES = ("ml_engine", "jobs")

#: Copied from old_repo, then deliberately changed for S3-only operation. Not hash-tracked.
ADAPTED = {
    "jobs/_common/runtime.py": (
        "build_store returns S3ObjectStore; --artifact-base help text; stage_image_uri(); "
        "training_job_name() (TRAINING_JOB_NAME, recorded per CV fold, CV Req 9.7)"
    ),
    "jobs/training/main.py": (
        "additive: optional plugin_factory on run_training and --checkpoint-dir for Spot "
        "checkpointing (design C4); default path is the inherited get_plugin; writes a "
        "threshold analysis for binary classifiers; CV_Mode dispatch to jobs/training/cv.py "
        "(optional checkpoint_dir for per-fit sub-directories) and optional details on "
        "write_model_failure (CV design C5, C9). grid-search: _name_for_spec stamps a grid "
        "point's '<base>__<hash>' spec name onto the metadata so each point ranks/lists as its "
        "own version; a plain name is unchanged. single-split path otherwise unchanged. "
        "ensembles-and-learners: an ensemble name resolves to an EnsemblePlugin via resolve_plugin "
        "(from the frozen spec) instead of the checkpointing factory"
    ),
    "ml_engine/io/__init__.py": "re-exports: local names removed, S3 names added",
    "ml_engine/io/layout.py": (
        "additive: compute/execution/checkpoint/spend + business-context/leakage-phase paths "
        "appended; cross-validation paths (fold assignment, development/test datasets, fold "
        "preprocessors, per-fold results, OOF, test evaluation) appended and two entries "
        "appended to RUN_OUTPUT_FILES (Req 4.1, 5.5, 10.6, 12.4); control-ui-api adds the "
        "model_version_index path (model_versions/index.json) and the validations/<id>/ paths "
        "(request/report/failure/input-dir); training-data-treatment adds the OutlierTreatment "
        "artifact paths (single-split + per-fold); free-form-experiments adds the "
        "free_form/<run_id>/ tree (request/result/failure/script/input/output/leakage/"
        "requirements/logs/model); ml-assistant adds AGENT_NAMESPACE, owner_key and AgentLayout "
        "(session, tool-result, in-flight, owner-index and audit paths under ml-factory/agent/); "
        "inherited paths unchanged"
    ),
    "ml_engine/contracts/config.py": (
        "additive: SplitConfig.evaluation_mode, .cv, .test_fraction (excluded at default). "
        "Req 1.1, 1.6, 3.1 put CV in the split configuration; defaults reproduce single split "
        "(Req 12.2). Also ExperimentConfig.requirements (RequirementsSpec | None, excluded at "
        "default) for the per-experiment package install (stock-image migration B10); None "
        "reproduces single-image configs byte-identically. training-data-treatment: "
        "ExperimentConfig.treatment (TreatmentConfig | None, excluded at default) freezes the "
        "approved outlier treatment; None/none reproduces today's chain byte-identically. "
        "ensembles-and-learners: ExperimentConfig.ensembles (list[EnsembleSpec], excluded when "
        "empty) freezes the approved voting/stacking ensembles; empty reproduces today"
    ),
    "ml_engine/contracts/preparation.py": (
        "additive: PreparationReport.cross_validation, .test_fold (excluded at default). "
        "Req 12.2; per-fold row counts for audit. training-data-treatment: optional "
        "applied_treatment (list[AppliedTreatment], excluded at default) records what outlier "
        "handling actually did per fold for audit (Req 6.6)"
    ),
    "ml_engine/contracts/model.py": (
        "additive: ModelMetadata.cross_validation (excluded at default). Req 8.3, 8.4 "
        "training record. ensembles-and-learners: ModelMetadata.ensemble_lineage "
        "(EnsembleLineage | None, excluded when unset) records an ensemble's base-learner "
        "versions (Req 6.4); absent for a single learner"
    ),
    "ml_engine/contracts/comparison.py": (
        "additive: ModelComparisonEntry.cv, .test_metrics; ComparisonReport.cross_validation, "
        ".test_evaluation (excluded at default). Req 7, 11.1, 12.2"
    ),
    "jobs/preprocessing/main.py": (
        "additive: dispatch to jobs/preprocessing/cv.py when CV or Test_Fold (after the "
        "inherited target drop and derived features); inherited body untouched for single "
        "split (Req 7.7, 13.6)"
    ),
    "jobs/evaluation/main.py": (
        "additive: one hook to jobs/evaluation/cv.py between build_comparison and "
        "write_model(comparison) when CV or a Test_Fold (CV ranking, CV_Summary, "
        "Test_Evaluation; CV design C6); single split without a Test_Fold is untouched so the "
        "comparison stays byte-identical (Req 7.7)"
    ),
    "ml_engine/io/object_store.py": (
        "LocalObjectStore and file:// handling removed; additive: Protocol.delete, copy_object "
        "uses a store's native server-side copy when it has one"
    ),
    "ml_engine/io/uri.py": "file:// helpers removed, s3:// parsing added",
    "ml_engine/splitting/strategies.py": (
        "additive: GroupSplit and TemporalSplit registered beside the inherited random and "
        "stratified strategies (the module docstring's planned extension point); inherited "
        "strategies and resolve_strategy unchanged"
    ),
    "ml_engine/contracts/execution.py": (
        "RESUMABLE_MODELS = frozenset() (stock-image migration, decision B4): no model resumes "
        "from a Spot checkpoint; every model is restart-on-interruption. The constant is kept "
        "(now empty) so the orchestrator and jobs/training/checkpointing.py imports resolve; "
        "Managed Spot is unchanged. Also adds WorkflowKind.VALIDATION to SIDE_WORKFLOWS for the "
        "control-ui-api validation side job (never touches experiment state). "
        "free-form-experiments: adds WorkflowKind.FREE_FORM to SIDE_WORKFLOWS (a user-approved "
        "script side job that never changes managed status). The rest of the "
        "file is the new execution/compute/spend contracts, unchanged"
    ),
    "ml_engine/models/registry.py": (
        "stock-image migration (decision B2/B10): CatBoost dropped from the import and the "
        "registration loop so the built-in set matches the stock AWS XGBoost image (sklearn + "
        "XGBoost); CatBoost is reachable via per-experiment requirements. "
        "ml_engine/models/catboost_plugin.py is kept on disk (unimported) for re-registration "
        "by the open catalogue. register/all_plugins unchanged. grid-search: add "
        "base_model_name/grid_point_id and make get_plugin resolve a '<base>__<hash>' grid-point "
        "name to its base plugin (single names unaffected). ensembles-and-learners: register the "
        "built-in bagging/boosting plugins (HistGradientBoosting/AdaBoost/Bagging/ExtraTrees) and "
        "the LightGBM installed-learner plugin; add is_ensemble_name/build_ensemble_plugin/"
        "resolve_plugin so an 'ensemble_<type>__<hash>' name resolves to an EnsemblePlugin built "
        "from the frozen EnsembleSpec (get_plugin stays pure for plain/grid names)"
    ),
    "ml_engine/models/base.py": (
        "grid-search (design C2): add parameter_space() (empty default) and validate_grid_value() "
        "so grid values can be validated before submission; param resolution is unchanged"
    ),
    "ml_engine/models/linear.py": (
        "grid-search: declare parameter_space on logistic_regression and elastic_net; defaults "
        "and build unchanged"
    ),
    "ml_engine/models/random_forest.py": (
        "grid-search: declare parameter_space on the random-forest base; defaults and build "
        "unchanged"
    ),
    "ml_engine/models/xgboost_plugin.py": (
        "grid-search: declare parameter_space on the xgboost base; defaults and build unchanged"
    ),
    "jobs/training/checkpointing.py": (
        "stock-image migration (decision B4): checkpointing_factory is now a no-op that returns "
        "the plain plugin (no read, no write), since Spot checkpoint/resume is dropped and "
        "RESUMABLE_MODELS is empty. The module and its public symbols are kept so the training "
        "job's import resolves; CV fold-level persistence in jobs/training/cv.py is unaffected"
    ),
}

#: Written for ML Factory 2.0; no old_repo counterpart. Not hash-tracked.
NEW = {
    "jobs/derived_profiling/__init__.py": "package marker",
    "jobs/derived_profiling/main.py": (
        "profile approved derived features on SageMaker, not in the control plane"
    ),
    "ml_engine/contracts/business_context.py": "plain-language business context (req 11)",
    "ml_engine/contracts/leakage_investigation.py": "human-controlled leakage warnings (req 12)",
    "ml_engine/contracts/thresholds.py": "decision-threshold analysis contract",
    "ml_engine/contracts/holdout.py": "evaluation of trained models on a separate dataset",
    "ml_engine/contracts/environment.py": "runtime environment captured inside a job",
    "jobs/_common/environment.py": "capture + write the job environment and requirements.txt",
    "jobs/holdout_evaluation/__init__.py": "package marker",
    "jobs/holdout_evaluation/main.py": "score trained models on a separate dataset (no refit)",
    "ml_engine/evaluation/thresholds.py": "precision/recall/F1 vs threshold (pure)",
    "ml_engine/contracts/report.py": "report document contracts (req 14)",
    "ml_engine/contracts/tracking.py": "status + failure-diagnosis contracts (design C5)",
    "ml_engine/io/s3.py": "S3ObjectStore (design C1, DD-2)",
    "ml_engine/leakage/investigation.py": "pre/post-training leakage phase builders (req 12)",
    "ml_engine/reporting/documents.py": "report document builders (req 14)",
    "ml_engine/reporting/render.py": "HTML renderer for reports (req 14)",
    "jobs/ablation/__init__.py": "package marker",
    "jobs/ablation/main.py": "retrain one model without each candidate feature (req 12, 13.5)",
    "jobs/scoring/__init__.py": "package marker",
    "jobs/scoring/main.py": "batch scoring of a new file with a trained model (no endpoint)",
    "ml_engine/contracts/explain.py": "permutation-importance and ablation contracts",
    "ml_engine/contracts/runs.py": "archived training-run manifest",
    "ml_engine/contracts/scoring.py": "batch scoring request/report contracts",
    "ml_engine/evaluation/permutation.py": "permutation importance per input column (pure)",
    "ml_engine/contracts/cv.py": "cross-validation and Test_Fold contracts (CV design)",
    "ml_engine/contracts/requirements.py": (
        "per-experiment pip requirements contract + well-formedness validation "
        "(stock-image migration B10/B13)"
    ),
    "ml_engine/contracts/model_version.py": (
        "Model_Version handle: a listable, independently-loadable pointer to a trained model in "
        "a run (control-ui-api design C4). free-form-experiments: ModelVersion.source gains "
        "'free_form' provenance (additive)"
    ),
    "ml_engine/contracts/grid.py": (
        "grid-search contracts: ParameterSpec, ParameterGrid, GridPoint, GridPlan (grid-search "
        "design C2/C3)"
    ),
    "ml_engine/models/grid.py": (
        "pure grid expansion: ParameterGrid -> GridPoints via the per-parameter cross-product "
        "(grid-search design C4)"
    ),
    "ml_engine/contracts/validation.py": (
        "validation request/report contracts for scoring model versions on a labeled dataset "
        "(control-ui-api design C5)"
    ),
    "jobs/validation/__init__.py": "package marker",
    "jobs/validation/main.py": (
        "validation job: score chosen model versions on a labeled dataset, keeping labels for "
        "metrics and a fresh threshold analysis (control-ui-api design C5). free-form-experiments: "
        "_resolve_versions also enumerates free-form run models so a free_form:<run>:<name> "
        "version validates through the same flow (Req 7.1)"
    ),
    "ml_engine/splitting/folds.py": (
        "Fold_Planner: Test_Fold carve-out and K-fold assignment (CV design C2); reuses "
        "strategies.py unchanged"
    ),
    "ml_engine/preprocessing/folds.py": (
        "fold-aware preprocessing fits: per-fold training rows + Development_Set (CV design C3); "
        "pipeline.py unchanged. training-data-treatment: fit + apply the per-fold OutlierTreatment "
        "on each fold's training rows before FeaturePipeline.fit (design C3); none is a no-op"
    ),
    "ml_engine/contracts/treatment.py": (
        "training-time data-treatment contracts: OutlierDetector/OutlierHandling enums, "
        "TreatmentConfig, OutlierBounds, OutlierProposal, AppliedTreatment "
        "(training-data-treatment design C1)"
    ),
    "ml_engine/preprocessing/outliers.py": (
        "the pure outlier engine: OutlierTreatment.fit/apply/bounds/report, composed around "
        "FeaturePipeline (training-data-treatment design C2); none is a pass-through"
    ),
    "ml_engine/contracts/ensemble.py": (
        "ensemble contracts: EnsembleType/VotingRule enums, BaseLearnerSpec, MetaLearnerSpec, "
        "EnsembleSpec (with a stable ensemble_id), EnsembleLineage (ensembles-and-learners C3)"
    ),
    "ml_engine/models/boosting.py": (
        "built-in boosting learners: HistGradientBoosting + AdaBoost (classifier + regressor), "
        "scikit-learn only (ensembles-and-learners C1)"
    ),
    "ml_engine/models/bagging_trees.py": (
        "built-in bagging learners: Bagging + ExtraTrees (classifier + regressor), scikit-learn "
        "only (ensembles-and-learners C1)"
    ),
    "ml_engine/models/lightgbm_plugin.py": (
        "LightGBM installed-learner plugin with a lazy import and is_available() gate; reachable "
        "via the per-experiment requirements install (ensembles-and-learners C2)"
    ),
    "ml_engine/models/ensemble.py": (
        "the EnsemblePlugin: assembles sklearn Voting/Stacking over base plugins' estimators from "
        "a frozen EnsembleSpec; trains through the unchanged train_model path (ensembles C4)"
    ),
    "ml_engine/contracts/free_form.py": (
        "free-form experiment contracts: MetricLabel, ScriptRevision, ApprovedScript, "
        "DataContextAnswers, FreeFormRequest, FreeFormMetric, FreeFormResult "
        "(free-form-experiments design C1)"
    ),
    "jobs/free_form/__init__.py": "package marker (free-form-experiments)",
    "jobs/free_form/main.py": (
        "the free-form job: run a content-hash-verified Approved_Script as a Processing job, "
        "capture + label results, register a Model_Version when a predict entry point is exposed "
        "(free-form-experiments design C3)"
    ),
    "ml_engine/sdk/__init__.py": (
        "the Platform_SDK façade: SDK_VERSION + splits/metrics/results re-exporting the engine's "
        "own pure code (free-form-experiments design C2)"
    ),
    "ml_engine/sdk/splits.py": (
        "SDK splitters: re-export managed strategies + temporal_holdout / rolling_origin_split "
        "with train-before-eval guarantees (free-form-experiments design C2, Req 5.1/5.4)"
    ),
    "ml_engine/sdk/metrics.py": (
        "SDK metrics: re-export evaluate + forecasting metrics with a seasonal-naive baseline "
        "(free-form-experiments design C2, Req 5.2/5.5)"
    ),
    "ml_engine/sdk/results.py": (
        "SDK results writer: a metric written through it is labelled 'engine' "
        "(free-form-experiments design C2, Req 4.2)"
    ),
    "jobs/preprocessing/cv.py": (
        "preparation with a Fold_Assignment: CV folds and/or Test_Fold (CV design C4)"
    ),
    "jobs/training/cv.py": (
        "CV_Mode training: fold loop with per-fold persistence/resume, OOF, refit, OOF "
        "thresholds and permutation aggregate (CV design C5). grid-search: stamp the grid "
        "point's spec name onto the refit metadata via _name_for_spec (same as the single-split "
        "path). training-data-treatment: apply the per-fold OutlierTreatment to a fold's training "
        "rows before transform/fit, and the development-level treatment at refit. "
        "ensembles-and-learners: _fit_plugin resolves an ensemble name to an EnsemblePlugin from "
        "the frozen spec"
    ),
    "ml_engine/reporting/cv.py": (
        "CV aggregation, ranking and instability/outlier warnings, pure (CV design C6)"
    ),
    "jobs/evaluation/cv.py": (
        "CV_Mode evaluation: CV_Summary per Run, CV ranking and the Test_Evaluation (CV design C6)"
    ),
    "ml_engine/reasoning/cv_context.py": (
        "cross_validation reasoning context built from the comparison only (CV design C7, Req 11.4)"
    ),
    "ml_engine/contracts/agent.py": (
        "versioned ML Assistant contracts: sessions, approvals, limits, tool-call records, "
        "final answers, audit records, args_hash/idempotency_key (ml-assistant design C1)"
    ),
    "ml_engine/reasoning/numbers.py": (
        "drop_unverified_numbers: remove LLM sentences citing numbers absent from the context "
        "(CV design C7, Req 11.5, D14)"
    ),
}


def digest(path: Path) -> str:
    text = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()


def tracked_files(root: Path = ROOT) -> list[str]:
    files = []
    for package in PACKAGES:
        for path in sorted((root / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            if relative not in ADAPTED and relative not in NEW:
                files.append(relative)
    return files


def build() -> dict:
    return {
        "source": "old_repo",
        "note": "Regenerate only after a reviewed, deliberate change: make manifest",
        "adapted": ADAPTED,
        "new": NEW,
        "files": {relative: digest(ROOT / relative) for relative in tracked_files()},
    }


def verify_against(old_root: Path) -> list[str]:
    problems = []
    for relative in tracked_files():
        original = old_root / relative
        if not original.is_file():
            problems.append(f"{relative}: no counterpart in {old_root}")
        elif digest(original) != digest(ROOT / relative):
            problems.append(f"{relative}: differs from {old_root}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="re-record the manifest")
    parser.add_argument("--verify-against", type=Path, help="path to old_repo")
    args = parser.parse_args(argv)

    if args.verify_against:
        problems = verify_against(args.verify_against.resolve())
        for problem in problems:
            print(problem)
        print(f"{len(tracked_files())} inherited files checked, {len(problems)} problems")
        if problems:
            return 1
    if args.write:
        MANIFEST.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
