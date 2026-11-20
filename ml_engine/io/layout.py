"""The one definition of the experiment artifact layout.

Never hardcode an artifact path anywhere else — import ``ExperimentLayout``.
Mirrored exactly by ``architecture.md`` §7.
"""

import hashlib
from dataclasses import dataclass

from ml_engine.io.uri import join_uri

ARTIFACT_NAMESPACE = "ml-factory/experiments"
#: Where a dataset uploaded through the UI lands before an experiment exists to own it.
UPLOAD_NAMESPACE = "ml-factory/uploads"
#: Inside an experiment: the dataset it was created from, under its original file name.
INPUT_DIR = "input"

DEFINITION_FILE = "experiment.json"
CONTROL_STATE_FILE = "state/control.json"
WORKFLOW_STATE_FILE = "state/workflow.json"
EDA_FILE = "eda/eda.json"
LEAKAGE_FILE = "eda/leakage.json"
# --- ML Factory 2.0: downloadable reports (design C8, requirement 14) ------------------------
REPORTS_DIR = "reports"
# --- ML Factory 2.0: business context + the two leakage-investigation phases (design C8) -----
BUSINESS_CONTEXT_FILE = "config/business_context.json"
LEAKAGE_PRE_TRAINING_FILE = "leakage/pre_training.json"
LEAKAGE_POST_TRAINING_FILE = "leakage/post_training.json"
EXPERIMENT_CONFIG_FILE = "config/experiment_config.json"
SELECTED_FEATURES_FILE = "config/selected_features.json"
FEATURE_PROPOSALS_FILE = "config/feature_proposals.json"
#: Per-derived-column profiles, computed when a person approves derived features. Row-wise and
#: unfitted, so profiling them before the split is leak-safe (see architecture.md AD-14).
DERIVED_PROFILE_FILE = "config/derived_profiles.json"
TRAINING_CONFIG_FILE = "config/training_config.json"
DATA_DICTIONARY_FILE = "config/data_dictionary.json"
PREPARATION_FILE = "validation/validation.json"
PREPROCESSOR_FILE_TEMPLATE = "preprocessing/preprocessor_{strategy}.joblib"
PREPROCESSING_METADATA_FILE = "preprocessing/preprocessing.json"
TRAIN_DATASET_FILE = "datasets/train.parquet"
VALIDATION_DATASET_FILE = "datasets/validation.parquet"
COMPARISON_FILE = "comparison/comparison.json"
SUMMARY_FILE = "report_data/experiment_summary.json"
REASONING_FILE = "reasoning/reasoning.json"
FAILURE_FILE = "failure.json"
# --- ML Factory 2.0 additions (SageMaker execution); nothing above this line changed ---------
#: Per-experiment compute choices (instance types, On-Demand opt-in).
COMPUTE_FILE = "config/compute.json"
#: One record per workflow execution, written only by the orchestrator.
EXECUTIONS_DIR = "executions"
#: SageMaker's own output (model.tar.gz, failure file) — not ML Factory artifacts.
SAGEMAKER_OUTPUT_DIR = "sagemaker/output"
SAGEMAKER_CHECKPOINT_DIR = "sagemaker/checkpoints"
#: Spend ledgers: one object per submitted job, written once, at submission.
EXPERIMENT_SPEND_DIR = "spend"
SPEND_NAMESPACE = "ml-factory/spend"
#: Earlier training runs, moved aside when an experiment is retrained.
RUNS_DIR = "runs"
ABLATION_FILE = "leakage/ablation.json"
# --- ML Factory 2.0: cross-validation and Test_Fold ------------------------------------------
#: Row -> Test_Fold / Training_Only / CV_Fold. Written only by the preparation job.
FOLD_ASSIGNMENT_FILE = "validation/fold_assignment.json"
#: CV_Mode: every non-Test_Fold row, in ascending row-id order.
DEVELOPMENT_DATASET_FILE = "datasets/development.parquet"
TEST_DATASET_FILE = "datasets/test.parquet"
TEST_EVALUATION_FILE = "comparison/test_evaluation.json"
FOLD_PREPROCESSOR_TEMPLATE = "preprocessing/folds/fold_{fold}/preprocessor_{strategy}.joblib"
# --- ML Factory 2.0: training-time data treatment (outlier handling) -------------------------
#: Single-split: the fitted OutlierTreatment applied to the training rows before the pipeline fit.
OUTLIER_TREATMENT_FILE = "preprocessing/outliers.joblib"
#: CV_Mode: one fitted OutlierTreatment per fold, applied at train time (jobs/training/cv.py).
FOLD_OUTLIER_TREATMENT_TEMPLATE = "preprocessing/folds/fold_{fold}/outliers.joblib"
#: Per-run outputs: archived and cleared before a retrain so nothing stale is read as new.
RUN_OUTPUT_PREFIXES = ("models", "comparison", "report_data", "validation", "preprocessing")
RUN_OUTPUT_FILES = (
    LEAKAGE_POST_TRAINING_FILE,
    ABLATION_FILE,
    "leakage/ablation_failure.json",
    "reasoning/reasoning.json",
    "reports/model_comparison.json",
    "reports/model_comparison.html",
    "reports/model_comparison.pdf",
    "environment/environment.json",
    "environment/requirements.txt",
    DEVELOPMENT_DATASET_FILE,
    TEST_DATASET_FILE,
)


@dataclass(frozen=True, slots=True)
class ExperimentLayout:
    """Resolves every artifact location for one experiment.

    ``base`` is an ``s3://bucket/ml-factory/experiments/<id>`` URI. Any ``ObjectStore`` can
    resolve it, so the same layout applies to a SageMaker job, the control plane, a notebook
    driving the engine directly, and the in-memory store used by the tests.
    """

    base: str

    @classmethod
    def for_experiment(cls, root: str, experiment_id: str) -> "ExperimentLayout":
        """``root`` is the artifact bucket URI."""
        return cls(base=join_uri(root, ARTIFACT_NAMESPACE, experiment_id))

    @staticmethod
    def experiments_prefix(root: str) -> str:
        """Where every experiment under an artifact root lives. Used to list them."""
        return join_uri(root, ARTIFACT_NAMESPACE) + "/"

    @staticmethod
    def uploads_prefix(root: str) -> str:
        """The upload inbox under an artifact root. Always an approved dataset location."""
        return join_uri(root, UPLOAD_NAMESPACE) + "/"

    @staticmethod
    def upload_object(root: str, folder: str, filename: str) -> str:
        """One folder per upload, holding the file under the name the user uploaded it with."""
        return join_uri(root, UPLOAD_NAMESPACE, folder, filename)

    def path(self, *parts: str) -> str:
        return join_uri(self.base, *parts)

    def input_dataset(self, filename: str) -> str:
        """The experiment's own copy of its dataset, keeping the original file name."""
        return self.path(INPUT_DIR, filename)

    @property
    def feature_proposals(self) -> str:
        return self.path(FEATURE_PROPOSALS_FILE)

    @property
    def derived_profiles(self) -> str:
        return self.path(DERIVED_PROFILE_FILE)

    @property
    def definition(self) -> str:
        return self.path(DEFINITION_FILE)

    @property
    def control_state(self) -> str:
        return self.path(CONTROL_STATE_FILE)

    @property
    def workflow_state(self) -> str:
        return self.path(WORKFLOW_STATE_FILE)

    @property
    def eda(self) -> str:
        return self.path(EDA_FILE)

    @property
    def leakage(self) -> str:
        return self.path(LEAKAGE_FILE)

    @property
    def business_context(self) -> str:
        return self.path(BUSINESS_CONTEXT_FILE)

    @property
    def leakage_pre_training(self) -> str:
        return self.path(LEAKAGE_PRE_TRAINING_FILE)

    @property
    def leakage_post_training(self) -> str:
        return self.path(LEAKAGE_POST_TRAINING_FILE)

    @property
    def experiment_config(self) -> str:
        return self.path(EXPERIMENT_CONFIG_FILE)

    @property
    def training_config(self) -> str:
        return self.path(TRAINING_CONFIG_FILE)

    @property
    def selected_features(self) -> str:
        return self.path(SELECTED_FEATURES_FILE)

    @property
    def data_dictionary(self) -> str:
        return self.path(DATA_DICTIONARY_FILE)

    @property
    def preparation(self) -> str:
        return self.path(PREPARATION_FILE)

    def preprocessor(self, strategy: str) -> str:
        """One fitted pipeline per preprocessing strategy; both fit on the train fold only."""
        return self.path(PREPROCESSOR_FILE_TEMPLATE.format(strategy=strategy))

    @property
    def preprocessing_metadata(self) -> str:
        return self.path(PREPROCESSING_METADATA_FILE)

    @property
    def train_dataset(self) -> str:
        return self.path(TRAIN_DATASET_FILE)

    @property
    def validation_dataset(self) -> str:
        return self.path(VALIDATION_DATASET_FILE)

    @property
    def comparison(self) -> str:
        return self.path(COMPARISON_FILE)

    @property
    def summary(self) -> str:
        return self.path(SUMMARY_FILE)

    @property
    def reasoning(self) -> str:
        return self.path(REASONING_FILE)

    @property
    def models_prefix(self) -> str:
        return self.path("models")

    def model_dir(self, model_name: str) -> str:
        return self.path("models", model_name)

    def model_artifact(self, model_name: str) -> str:
        return self.path("models", model_name, "model.joblib")

    def model_metadata(self, model_name: str) -> str:
        return self.path("models", model_name, "metadata.json")

    def model_failure(self, model_name: str) -> str:
        return self.path("models", model_name, FAILURE_FILE)

    def model_thresholds(self, model_name: str) -> str:
        """ML Factory 2.0: decision-threshold analysis (binary classification only)."""
        return self.path("models", model_name, "thresholds.json")

    # --- ML Factory 2.0: the environment the models were trained in -------------------------
    @property
    def environment(self) -> str:
        return self.path("environment", "environment.json")

    @property
    def requirements(self) -> str:
        return self.path("environment", "requirements.txt")

    # --- ML Factory 2.0: evaluations on a separate dataset ----------------------------------
    @property
    def evaluations_prefix(self) -> str:
        return self.path("evaluations") + "/"

    def evaluation_request(self, evaluation_id: str) -> str:
        return self.path("evaluations", evaluation_id, "request.json")

    def evaluation_report(self, evaluation_id: str) -> str:
        return self.path("evaluations", evaluation_id, "holdout.json")

    def evaluation_failure(self, evaluation_id: str) -> str:
        return self.path("evaluations", evaluation_id, FAILURE_FILE)

    def report(self, name: str) -> str:
        return self.path(REPORTS_DIR, name)

    def report_document(self, kind: str, extension: str) -> str:
        """A downloadable report, e.g. ``reports/eda.json`` or ``reports/model_comparison.pdf``."""
        return self.path(REPORTS_DIR, f"{kind}.{extension}")

    # --- ML Factory 2.0 additions ---------------------------------------------------------
    @staticmethod
    def monthly_spend_prefix(root: str, month: str) -> str:
        """Platform-wide spend ledger for one ``YYYY-MM``, listed to enforce the monthly cap."""
        return join_uri(root, SPEND_NAMESPACE, month) + "/"

    @property
    def compute(self) -> str:
        return self.path(COMPUTE_FILE)

    @property
    def executions_prefix(self) -> str:
        return self.path(EXECUTIONS_DIR) + "/"

    def execution(self, execution_id: str) -> str:
        return self.path(EXECUTIONS_DIR, f"{execution_id}.json")

    def observation(self, execution_id: str, job_name: str) -> str:
        """Latest status observation for one job. Written only by the status monitor."""
        return self.path(EXECUTIONS_DIR, execution_id, "observations", f"{job_name}.json")

    def observations_prefix(self, execution_id: str) -> str:
        return self.path(EXECUTIONS_DIR, execution_id, "observations") + "/"

    def diagnosis(self, execution_id: str, job_name: str) -> str:
        """Failure diagnosis for one job. Written only by the diagnoser."""
        return self.path(EXECUTIONS_DIR, execution_id, "diagnosis", f"{job_name}.json")

    @property
    def sagemaker_output(self) -> str:
        return self.path(SAGEMAKER_OUTPUT_DIR)

    def checkpoint(self, model_name: str, run: str | None = None) -> str:
        """Spot checkpoint prefix. Scoped per run so a retrain never resumes an older model."""
        if run:
            return self.path(SAGEMAKER_CHECKPOINT_DIR, run, model_name)
        return self.path(SAGEMAKER_CHECKPOINT_DIR, model_name)

    # --- ML Factory 2.0: archived earlier training runs ----------------------------------
    @property
    def runs_prefix(self) -> str:
        return self.path(RUNS_DIR) + "/"

    def run_archive(self, run_id: str) -> str:
        """Where an earlier run's results are moved before the experiment is retrained."""
        return self.path(RUNS_DIR, run_id)

    def run_manifest(self, run_id: str) -> str:
        return self.path(RUNS_DIR, run_id, "run.json")

    # --- ML Factory 2.0: model-version store (control-ui-api design C4) ------------------
    @property
    def model_version_index(self) -> str:
        """Optional cache of the enumerated Model_Versions; the per-model metadata is the truth."""
        return self.path("model_versions", "index.json")

    # --- ML Factory 2.0: post-training explanation and ablation --------------------------
    def model_permutation(self, model_name: str) -> str:
        """Permutation importance on the validation fold, per input column."""
        return self.path("models", model_name, "permutation.json")

    @property
    def ablation(self) -> str:
        return self.path(ABLATION_FILE)

    @property
    def ablation_failure(self) -> str:
        return self.path("leakage", "ablation_failure.json")

    @property
    def leakage_reasoning(self) -> str:
        """The LLM's pre-training leak reasoning (governed context), kept for traceability."""
        return self.path("reasoning", "leakage_reasoning.json")

    # --- ML Factory 2.0: batch scoring --------------------------------------------------
    @property
    def scoring_prefix(self) -> str:
        return self.path("scoring") + "/"

    def scoring_request(self, scoring_id: str) -> str:
        return self.path("scoring", scoring_id, "request.json")

    def scoring_report(self, scoring_id: str) -> str:
        return self.path("scoring", scoring_id, "scoring.json")

    def scoring_predictions(self, scoring_id: str) -> str:
        return self.path("scoring", scoring_id, "predictions.csv")

    def scoring_failure(self, scoring_id: str) -> str:
        return self.path("scoring", scoring_id, FAILURE_FILE)

    # --- ML Factory 2.0: validation of model versions on a labeled dataset (control-ui-api) --
    # Under ``validations/`` (not the CV ``validation/`` fold dir, and not in RUN_OUTPUT_PREFIXES)
    # so a validation survives a retrain, like a holdout evaluation.
    @property
    def validations_prefix(self) -> str:
        return self.path("validations") + "/"

    def validation_request(self, validation_id: str) -> str:
        return self.path("validations", validation_id, "request.json")

    def validation_report(self, validation_id: str) -> str:
        return self.path("validations", validation_id, "validation.json")

    def validation_failure(self, validation_id: str) -> str:
        return self.path("validations", validation_id, FAILURE_FILE)

    def validation_input_dir(self, validation_id: str) -> str:
        return self.path("validations", validation_id, INPUT_DIR)

    # --- ML Factory 2.0: free-form experiments (free-form-experiments design C7) -------------
    # Under ``free_form/`` (survives a retrain, like validations; not in RUN_OUTPUT_PREFIXES) so a
    # free-form run and its saved model/code outlive a managed retrain.
    @property
    def free_form_prefix(self) -> str:
        return self.path("free_form") + "/"

    def free_form_request(self, run_id: str) -> str:
        return self.path("free_form", run_id, "request.json")

    def free_form_result(self, run_id: str) -> str:
        return self.path("free_form", run_id, "result.json")

    def free_form_failure(self, run_id: str) -> str:
        return self.path("free_form", run_id, FAILURE_FILE)

    def free_form_script(self, run_id: str) -> str:
        """The pinned Approved_Script bytes the run executes (content-hash verified)."""
        return self.path("free_form", run_id, "script.py")

    def free_form_input_dir(self, run_id: str) -> str:
        return self.path("free_form", run_id, INPUT_DIR)

    def free_form_output_dir(self, run_id: str) -> str:
        """Where the script writes its results and model; isolated per run (Req 1.7)."""
        return self.path("free_form", run_id, "output")

    def free_form_leakage(self, run_id: str) -> str:
        return self.path("free_form", run_id, "leakage.json")

    def free_form_draft(self, draft_id: str) -> str:
        """A drafting session (data-context answers + retained Script_Draft revisions)."""
        return self.path("free_form", "drafts", f"{draft_id}.json")

    @property
    def free_form_drafts_prefix(self) -> str:
        return self.path("free_form", "drafts") + "/"

    def free_form_data_context_defaults(self, dataset_key: str) -> str:
        """Saved data-context answers, reused as defaults for later runs on the same dataset."""
        return self.path("free_form", "data_context", f"{dataset_key}.json")

    def free_form_requirements(self, run_id: str) -> str:
        """The resolved ``pip freeze`` captured at run time (B11)."""
        return self.path("free_form", run_id, "requirements.txt")

    def free_form_logs(self, run_id: str) -> str:
        return self.path("free_form", run_id, "logs.txt")

    def free_form_models_prefix(self, run_id: str) -> str:
        """The models/ prefix of one free-form run, listed to enumerate its Model_Versions."""
        return self.path("free_form", run_id, "models") + "/"

    def free_form_model_dir(self, run_id: str, model_name: str) -> str:
        """A Model_Version-shaped tree for a run that exposes a Predict_Entry_Point (Req 6.3)."""
        return self.path("free_form", run_id, "models", model_name)

    def free_form_model_artifact(self, run_id: str, model_name: str) -> str:
        return self.path("free_form", run_id, "models", model_name, "model.joblib")

    def free_form_model_metadata(self, run_id: str, model_name: str) -> str:
        return self.path("free_form", run_id, "models", model_name, "metadata.json")

    # --- ML Factory 2.0: pinned copies of every dataset a stage reads ---------------------
    @property
    def input_dir(self) -> str:
        return self.path(INPUT_DIR)

    def evaluation_input_dir(self, evaluation_id: str) -> str:
        return self.path("evaluations", evaluation_id, INPUT_DIR)

    def scoring_input_dir(self, scoring_id: str) -> str:
        return self.path("scoring", scoring_id, INPUT_DIR)

    @property
    def spend_prefix(self) -> str:
        return self.path(EXPERIMENT_SPEND_DIR) + "/"

    # --- ML Factory 2.0: cross-validation and Test_Fold ---------------------------------
    @property
    def fold_assignment(self) -> str:
        return self.path(FOLD_ASSIGNMENT_FILE)

    @property
    def development_dataset(self) -> str:
        return self.path(DEVELOPMENT_DATASET_FILE)

    @property
    def test_dataset(self) -> str:
        return self.path(TEST_DATASET_FILE)

    def fold_preprocessor(self, fold: int, strategy: str) -> str:
        """A pipeline fit on one CV_Fold's training rows only."""
        return self.path(FOLD_PREPROCESSOR_TEMPLATE.format(fold=fold, strategy=strategy))

    @property
    def outlier_treatment(self) -> str:
        """Single-split: the OutlierTreatment fit on the training rows (training-data-treatment)."""
        return self.path(OUTLIER_TREATMENT_FILE)

    def fold_outlier_treatment(self, fold: int) -> str:
        """CV_Mode: the OutlierTreatment fit on one CV_Fold's training rows (applied at train)."""
        return self.path(FOLD_OUTLIER_TREATMENT_TEMPLATE.format(fold=fold))

    def model_cv_dir(self, model_name: str) -> str:
        return self.path("models", model_name, "cv")

    def model_fold_result(self, model_name: str, fold: int) -> str:
        """Completion marker of one fold; a restarted job skips folds that have one."""
        return self.path("models", model_name, "cv", "folds", f"fold_{fold}.json")

    def model_fold_oof(self, model_name: str, fold: int) -> str:
        return self.path("models", model_name, "cv", "folds", f"fold_{fold}_oof.parquet")

    def model_fold_metrics(self, model_name: str) -> str:
        return self.path("models", model_name, "cv", "fold_metrics.json")

    def model_oof(self, model_name: str) -> str:
        """Manifest of the out-of-fold predictions (the versioned contract)."""
        return self.path("models", model_name, "cv", "oof.json")

    def model_oof_payload(self, model_name: str) -> str:
        return self.path("models", model_name, "cv", "oof.parquet")

    @property
    def test_evaluation(self) -> str:
        return self.path(TEST_EVALUATION_FILE)


# --- ML Assistant: agent sessions (ml-assistant design C2) -----------------------------------
#: Where agent sessions, their tool results, in-flight markers and audit objects live.
AGENT_NAMESPACE = "ml-factory/agent"
AGENT_SESSIONS_DIR = "sessions"
AGENT_OWNERS_DIR = "owners"


def owner_key(owner: str) -> str:
    """A path-safe, stable key for a user identity: the first 24 hex characters of its SHA-256."""
    return hashlib.sha256(owner.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class AgentLayout:
    """Resolves every location of one agent session.

    ``base`` is ``<root>/ml-factory/agent/sessions/<session_id>``.
    """

    base: str

    @classmethod
    def for_session(cls, root: str, session_id: str) -> "AgentLayout":
        return cls(base=join_uri(root, AGENT_NAMESPACE, AGENT_SESSIONS_DIR, session_id))

    @staticmethod
    def owners_prefix(root: str) -> str:
        """Where every owner's session index lives. Listed by the job watcher's discovery."""
        return join_uri(root, AGENT_NAMESPACE, AGENT_OWNERS_DIR) + "/"

    @staticmethod
    def owner_index_prefix(root: str, owner: str) -> str:
        """Where one owner's session index objects live. Listed to enumerate their sessions."""
        return join_uri(root, AGENT_NAMESPACE, AGENT_OWNERS_DIR, owner_key(owner)) + "/"

    @staticmethod
    def owner_index_object(root: str, owner: str, session_id: str) -> str:
        return join_uri(
            root, AGENT_NAMESPACE, AGENT_OWNERS_DIR, owner_key(owner), f"{session_id}.json"
        )

    @property
    def session(self) -> str:
        return join_uri(self.base, "session.json")

    def tool_result(self, tool_call_id: str) -> str:
        return join_uri(self.base, "results", f"{tool_call_id}.json")

    def inflight(self, idempotency_key: str) -> str:
        return join_uri(self.base, "inflight", f"{idempotency_key}.json")

    @property
    def audit_prefix(self) -> str:
        """Written by ``AuditLog`` when this session's base is its artifact prefix."""
        return join_uri(self.base, "audit") + "/"
