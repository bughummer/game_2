"""Control-plane configuration: the one module that reads the environment.

Every setting is ``ML_FACTORY_<NAME>`` in the environment (or passed explicitly in tests).
There is exactly one production topology — S3 artifacts, SageMaker execution — so there is no
"mode" setting; ``problems()`` reports what is missing for that topology.

Budget defaults are placeholders until the owner supplies real ceilings (design Q1); Spot wait
and the instance ceiling likewise (Q2).
"""

from decimal import Decimal
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.governance import bedrock_region_mismatch
from ml_engine.io.uri import InvalidS3UriError, normalize_storage_root, parse_s3_uri

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICE_TABLE = ROOT / "infra" / "pricing" / "sagemaker_prices.json"

#: Accelerated families. CPU only until benchmarking justifies otherwise (requirement 4.5).
GPU_FAMILY_PREFIXES = ("ml.p", "ml.g", "ml.inf", "ml.trn", "ml.dl")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML_FACTORY_", extra="forbid")

    # --- AWS identity -------------------------------------------------------------------
    aws_region: str = "us-east-1"
    aws_profile: str | None = None
    aws_role_arn: str | None = Field(
        default=None, description="Control-plane role to assume; refreshed automatically."
    )
    https_proxy: str | None = None

    # --- storage ------------------------------------------------------------------------
    artifact_root: str = ""
    allowed_dataset_prefixes: list[str] = Field(default_factory=list)
    s3_sse: str = "AES256"
    s3_kms_key_id: str | None = None
    max_dataset_upload_bytes: int = Field(default=500 * 1024 * 1024, ge=1)
    dataset_preview_rows: int = Field(default=50, ge=1, le=1000)
    presigned_url_ttl_seconds: int = Field(default=3600, ge=60, le=86_400)
    experiment_list_limit: int = Field(default=200, ge=1, le=5000)

    # --- identity (dev stand-in until LibreChat auth is real) ---------------------------
    dev_identity: str | None = None

    # --- Bedrock + data governance (requirement 15.5, 16.4) -----------------------------
    bedrock_enabled: bool = False
    bedrock_model_id: str = ""
    bedrock_max_tokens: int = Field(default=4096, ge=256, le=32_000)
    #: What may reach Bedrock. Default: column names + association/importance stats + capped
    #: sample values. Full datasets are never sent regardless of this flag.
    bedrock_send_sample_values: bool = True
    bedrock_sample_values_per_column: int = Field(default=5, ge=0, le=50)
    bedrock_sample_value_max_chars: int = Field(default=64, ge=1, le=512)

    # --- limits / retention / audit (requirement 15.4) ----------------------------------
    audit_log_enabled: bool = True
    data_retention_days: int = Field(default=0, ge=0, description="0 = retain indefinitely.")

    # --- SageMaker ----------------------------------------------------------------------
    sagemaker_execution_role_arn: str = ""
    #: Optional override of the stage image URI (e.g. to pin a specific digest). When unset the
    #: control plane resolves the stock AWS XGBoost image for ``aws_region`` at submission
    #: (backend/aws/images.py). No longer required by ``problems()`` (stock-image migration).
    stage_image_uri: str = ""
    #: Framework version passed to ``image_uris.retrieve``; None lets the SDK pick its default.
    xgboost_image_version: str | None = None
    #: Per-experiment package install controls (Stage 2). Open by default (decision B13): the
    #: allowlist is OFF unless an operator turns it on; kept here so config is stable.
    package_install_allowlist: list[str] = Field(default_factory=list)
    enforce_package_allowlist: bool = False
    training_instance_type: str = "ml.m5.large"
    processing_instance_type: str = "ml.m5.large"
    allowed_instance_types: list[str] = Field(
        default_factory=lambda: ["ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge"],
        description="The ceiling: nothing outside this list is ever submitted.",
    )
    volume_size_gb: int = Field(default=30, ge=1, le=512)
    volume_kms_key_id: str | None = None
    training_max_runtime_seconds: int = Field(default=3600, ge=60)
    processing_max_runtime_seconds: int = Field(default=3600, ge=60)
    # --- free-form experiments ----------------------------------------------------------
    #: Instance types a free-form run may use beyond ``allowed_instance_types`` — this is where an
    #: operator opts GPU boxes in (Req 1.4). Empty by default, so free-form is CPU-only like the
    #: managed tier until an operator widens it.
    allowed_free_form_instance_types: list[str] = Field(default_factory=list)
    #: The runtime ceiling of one free-form run (Req 10.1, open q 3).
    free_form_max_runtime_seconds: int = Field(default=7200, ge=60)
    spot_enabled: bool = True
    spot_max_wait_seconds: int = Field(default=3600, ge=60)
    max_parallel_training_jobs: int = Field(
        default=5, ge=1, le=20, description="Per experiment; further models wait for a slot."
    )
    max_platform_training_jobs: int = Field(
        default=20, ge=1, le=500, description="All ML Factory training jobs across the account."
    )
    max_models_per_run: int = Field(default=20, ge=1, le=50)
    #: Copy each dataset into its experiment's own prefix at creation, so later edits to the
    #: source object can never change what the experiment trains on.
    pin_datasets: bool = True
    #: The built-in poller that advances workflows (backend/tracking/poller.py). 0 disables it —
    #: set 0 on every MCP replica but one, or when EventBridge drives the monitor instead.
    status_poll_interval_seconds: int = Field(default=30, ge=0, le=3600)
    status_rediscover_interval_seconds: int = Field(default=900, ge=60)
    training_retry_attempts: int | None = Field(
        default=None, ge=1, le=30, description="SageMaker RetryStrategy; None omits it."
    )
    vpc_subnet_ids: list[str] = Field(default_factory=list)
    vpc_security_group_ids: list[str] = Field(default_factory=list)

    # --- spend ------------------------------------------------------------------------
    # Budgets, pricing and cost guardrails are owned by the MLOps team (AWS Budgets, Cost
    # Explorer, the Canvas cost tooling). ML Factory's own budget guard is therefore OFF by
    # default: it does not price or cap jobs unless explicitly enabled. When enabled it enforces
    # the ceilings below and refuses to submit a job it cannot price (fail closed).
    enforce_budget: bool = False
    budget_per_experiment_usd: float = Field(default=25.0, gt=0)
    budget_per_month_usd: float = Field(default=500.0, gt=0)
    price_table_path: Path = DEFAULT_PRICE_TABLE
    require_verified_prices: bool = True

    # --- cross-validation and Test_Fold (ML Factory 2.0, CV design C1) ------------------
    # Thresholds are copied into the frozen ExperimentConfig at training start (the SageMaker
    # stages never read Settings), so the instability and outlier flags are reproducible from
    # the config alone.
    cv_default_folds: int = Field(
        default=5, ge=2, le=50, description="K when CV_Mode is requested without a value for K."
    )
    cv_max_folds: int = Field(
        default=10, ge=2, le=50, description="Upper bound of K; the lower bound is fixed at 2."
    )
    cv_max_fits: int = Field(
        default=60, ge=3, description="Ceiling on enabled models x (K + 1) fold fits."
    )
    cv_training_max_runtime_seconds: int | None = Field(
        default=None,
        gt=0,
        description="Runtime ceiling of a CV Training_Job; None scales training_max_runtime.",
    )
    default_test_fraction: float = Field(
        default=0.2, gt=0.0, lt=0.5, description="Test_Fold fraction when enabled without one."
    )
    cv_instability_std: float = Field(
        default=0.05, gt=0.0, description="Absolute std threshold of the primary metric."
    )
    cv_instability_cv: float = Field(
        default=0.10, gt=0.0, description="Coefficient-of-variation threshold."
    )
    cv_outlier_multiplier: float = Field(
        default=3.0, gt=0.0, description="Outlier fold: distance > this x the other folds' std."
    )
    cv_outlier_min_gap: float = Field(
        default=0.02, ge=0.0, description="Outlier fold: and > this absolute gap."
    )

    # --- ML Assistant (agent) -----------------------------------------------------------
    # No ge/le here on purpose: ``problems()`` reports name, range and value (Req 13.3).
    agent_enabled: bool = False
    #: Empty falls back to ``bedrock_model_id`` (see ``effective_agent_model_id``).
    agent_model_id: str = ""
    agent_max_output_tokens: int = 2048
    agent_converse_retries: int = 3
    agent_converse_backoff_seconds: float = 1.0
    agent_converse_read_timeout_seconds: int = 120
    agent_limit_max_converse_turns: int = 40
    agent_limit_max_tool_calls: int = 60
    agent_limit_max_costly_tool_calls: int = 10
    agent_limit_max_estimated_spend_usd: Decimal = Decimal("10.0")
    agent_limit_max_active_seconds: int = 7200
    agent_limit_max_repeated_identical_calls: int = 3
    agent_approval_ttl_seconds: int = 900
    agent_tool_result_max_bytes: int = 8192
    agent_untrusted_string_max_chars: int = 256
    agent_context_budget_tokens: int = 100_000
    agent_cli_poll_seconds: float = 1.0
    agent_web_poll_seconds: float = 1.0
    agent_watch_interval_seconds: int = 15
    agent_lease_seconds: int = 300
    agent_max_concurrent_sessions: int = 4
    agent_autopilot_allowed: bool = True
    agent_api_url: str = "http://localhost:8000"

    @property
    def effective_agent_model_id(self) -> str:
        """The agent's model: ``agent_model_id``, else ``bedrock_model_id``."""
        return self.agent_model_id or self.bedrock_model_id

    @model_validator(mode="after")
    def _cv_bounds(self) -> "Settings":
        if not 2 <= self.cv_default_folds <= self.cv_max_folds:
            raise ValueError(
                "cv_default_folds must satisfy 2 <= cv_default_folds <= cv_max_folds "
                f"(got {self.cv_default_folds}, cv_max_folds={self.cv_max_folds})"
            )
        if self.cv_max_fits < 3:
            raise ValueError(f"cv_max_fits must be >= 3 (got {self.cv_max_fits})")
        return self

    @field_validator("artifact_root")
    @classmethod
    def _root(cls, value: str) -> str:
        return normalize_storage_root(value)

    @field_validator("allowed_dataset_prefixes")
    @classmethod
    def _prefixes(cls, value: list[str]) -> list[str]:
        return [normalize_storage_root(v) for v in value if v.strip()]

    @field_validator("s3_sse")
    @classmethod
    def _sse(cls, value: str) -> str:
        if value not in {"AES256", "aws:kms"}:
            raise ValueError("s3_sse must be 'AES256' or 'aws:kms'")
        return value

    @field_validator("allowed_instance_types")
    @classmethod
    def _cpu_only(cls, value: list[str]) -> list[str]:
        for instance in value:
            if not instance.startswith("ml."):
                raise ValueError(f"{instance!r} is not a SageMaker instance type")
            if instance.startswith(GPU_FAMILY_PREFIXES):
                raise ValueError(f"{instance!r} is an accelerated instance; CPU only (req 4.5)")
        return value

    def problems(self) -> list[str]:
        """Everything that would stop the platform submitting a job. Empty means ready."""
        found: list[str] = []
        if not self.artifact_root:
            found.append("ML_FACTORY_ARTIFACT_ROOT is required (s3://bucket[/prefix])")
        else:
            try:
                parse_s3_uri(self.artifact_root)
            except InvalidS3UriError as error:
                found.append(str(error))
        if not self.sagemaker_execution_role_arn.startswith("arn:aws"):
            found.append("ML_FACTORY_SAGEMAKER_EXECUTION_ROLE_ARN must be an IAM role ARN")
        # stage_image_uri is optional: when unset the control plane resolves the stock AWS
        # XGBoost image at submission (stock-image migration, requirement 1.1).
        for name in ("training_instance_type", "processing_instance_type"):
            if getattr(self, name) not in self.allowed_instance_types:
                found.append(f"{name} {getattr(self, name)!r} is outside allowed_instance_types")
        if self.s3_kms_key_id and self.s3_sse != "aws:kms":
            found.append("ML_FACTORY_S3_KMS_KEY_ID requires ML_FACTORY_S3_SSE=aws:kms")
        if self.bedrock_enabled:
            if not self.bedrock_model_id:
                found.append("ML_FACTORY_BEDROCK_MODEL_ID is required when Bedrock is enabled")
            else:
                mismatch = bedrock_region_mismatch(self.bedrock_model_id, self.aws_region)
                if mismatch:
                    found.append(mismatch)
        if self.spot_enabled and self.spot_max_wait_seconds < self.training_max_runtime_seconds:
            found.append(
                "ML_FACTORY_SPOT_MAX_WAIT_SECONDS must be >= "
                "ML_FACTORY_TRAINING_MAX_RUNTIME_SECONDS"
            )
        if bool(self.vpc_subnet_ids) != bool(self.vpc_security_group_ids):
            found.append("VPC config needs both subnet ids and security group ids")
        if self.enforce_budget and not self.price_table_path.is_file():
            found.append(f"price table not found: {self.price_table_path}")
        found.extend(self._agent_problems())
        return found

    def _agent_problems(self) -> list[str]:
        """Agent settings out of range, plus the model id and region checks when enabled."""
        found: list[str] = []
        for name, low, high in _AGENT_RANGES:
            value = getattr(self, name)
            if not low <= value <= high:
                found.append(
                    f"ML_FACTORY_{name.upper()} must be {_range_text(low, high)} (got {value})"
                )
        for name in _AGENT_POSITIVE_INTS:
            value = getattr(self, name)
            if value < 1:
                found.append(f"ML_FACTORY_{name.upper()} must be a positive integer (got {value})")
        spend = self.agent_limit_max_estimated_spend_usd
        if not spend.is_finite() or spend <= 0:
            found.append(
                "ML_FACTORY_AGENT_LIMIT_MAX_ESTIMATED_SPEND_USD must be greater than 0 "
                f"(got {spend})"
            )
        if self.agent_enabled:
            model_id = self.effective_agent_model_id
            if not model_id:
                found.append(
                    "ML_FACTORY_AGENT_MODEL_ID or ML_FACTORY_BEDROCK_MODEL_ID is required "
                    "when the agent is enabled"
                )
            else:
                mismatch = bedrock_region_mismatch(model_id, self.aws_region)
                if mismatch:
                    found.append(mismatch)
        return found


#: (field, low, high) inclusive ranges checked by ``Settings._agent_problems``.
_AGENT_RANGES: tuple[tuple[str, float, float], ...] = (
    ("agent_max_output_tokens", 1, 8192),
    ("agent_converse_retries", 0, 5),
    ("agent_converse_backoff_seconds", 0.0, 30.0),
    ("agent_converse_read_timeout_seconds", 10, 600),
    ("agent_approval_ttl_seconds", 60, 86_400),
    ("agent_tool_result_max_bytes", 1_024, 1_048_576),
    ("agent_untrusted_string_max_chars", 1, 100_000),
    ("agent_context_budget_tokens", 4_000, 1_000_000),
    ("agent_cli_poll_seconds", 0.5, 60),
    ("agent_web_poll_seconds", 0.5, 60),
    ("agent_watch_interval_seconds", 1, 60),
    ("agent_lease_seconds", 30, 3600),
    ("agent_max_concurrent_sessions", 1, 64),
)
_AGENT_POSITIVE_INTS: tuple[str, ...] = (
    "agent_limit_max_converse_turns",
    "agent_limit_max_tool_calls",
    "agent_limit_max_costly_tool_calls",
    "agent_limit_max_active_seconds",
    "agent_limit_max_repeated_identical_calls",
)


def _range_text(low: float, high: float) -> str:
    return f"between {_bound_text(low)} and {_bound_text(high)}"


def _bound_text(value: float) -> str:
    """A bound in full, never in exponent form: 1048576, 0.5, 0 (``:g`` gives 1.04858e+06)."""
    return str(int(value)) if float(value).is_integer() else str(value)
