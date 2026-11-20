"""Golden single-split artifacts captured before the cross-validation feature (task 1.1).

A Single_Split_Mode experiment is run through the real stages (``InlineOrchestrator``) over the
in-memory store, and five artifacts are kept under ``tests/fixtures/pre_cv/``. After masking the
values that legitimately change between runs or machines (timestamps, durations, the random
experiment id, dataset object metadata, the captured Python environment) the artifacts of a
later run must be byte-identical to these files (requirements 12.3, 13.6).

Regenerate only from pre-feature code, or after a reviewed, deliberate change::

    .venv\\Scripts\\python -m tests.support.golden --write
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_engine.io import ExperimentLayout, ObjectStore

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "pre_cv"
GOLDEN_MODELS = ["logistic_regression", "random_forest"]
#: The model whose ``metadata.json`` is kept.
GOLDEN_METADATA_MODEL = "logistic_regression"
GOLDEN_FEATURES = ["tenure_months", "monthly_charges", "contract", "is_business", "signup_date"]
TARGET = "churned"
MASK = "<masked>"
EXPERIMENT_ID_MASK = "<experiment_id>"

#: Fixture file name -> where the artifact lives in an experiment.
GOLDEN_FILES: dict[str, Callable[[ExperimentLayout], str]] = {
    "experiment_config.json": lambda layout: layout.experiment_config,
    "validation.json": lambda layout: layout.preparation,
    "metadata.json": lambda layout: layout.model_metadata(GOLDEN_METADATA_MODEL),
    "comparison.json": lambda layout: layout.comparison,
    "experiment_summary.json": lambda layout: layout.summary,
}

#: Keys whose values vary per run or per machine. Their presence is kept; the value is masked.
MASKED_KEYS = frozenset(
    {
        # timestamps
        "archived_at",
        "captured_at",
        "completed_at",
        "created_at",
        "decided_at",
        "failed_at",
        "generated_at",
        "last_modified",
        "started_at",
        "training_completed_at",
        "training_started_at",
        "updated_at",
        # durations
        "duration_seconds",
        "training_duration_seconds",
        # dataset object metadata (CSV line endings differ per OS)
        "etag",
        "size_bytes",
        "version_id",
        # the captured environment
        "package_versions",
        "platform",
        "python_version",
    }
)


#: Lists the service builds from a set, so their order follows Python's per-process string hash
#: seed. Sorted before comparison; their content is still compared exactly.
SORTED_KEYS = frozenset({"excluded_features"})


def golden_frame() -> pd.DataFrame:
    """The ``classification_frame`` fixture from ``tests/conftest.py``, reproduced verbatim."""
    rng = np.random.default_rng(42)
    rows = 400
    tenure = rng.integers(1, 60, rows)
    charges = rng.normal(70, 20, rows)
    contract = rng.choice(["monthly", "annual"], rows, p=[0.6, 0.4])
    risk = 0.05 * charges - 0.04 * tenure + np.where(contract == "monthly", 1.5, -1.0)
    target = (risk + rng.normal(0, 1.0, rows) > 2.0).astype(int)
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(rows)],
            "tenure_months": tenure,
            "monthly_charges": np.round(charges, 2),
            "contract": contract,
            "is_business": rng.choice([True, False], rows),
            "signup_date": pd.date_range("2021-01-01", periods=rows, freq="D"),
            "region_code": "EU",
            "notes": rng.choice(
                [
                    "customer provided a long free text comment about the service quality",
                    "the technician visit was handled professionally and on time",
                    "billing was unclear and the price increase was not communicated",
                ],
                rows,
            ),
            "churn_copy": target,
            "churned": target,
        }
    )
    frame.loc[rng.random(rows) < 0.4, "monthly_charges"] = np.nan
    return frame


def run_single_split_experiment(store: ObjectStore) -> ExperimentLayout:
    """Create, profile and train one Single_Split_Mode experiment with the real stages."""
    from backend.config import Settings
    from backend.repositories import ExperimentRepository
    from backend.schemas import TrainingRequest
    from backend.services import ExperimentService
    from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
    from tests.support.inline_orchestrator import InlineOrchestrator

    settings = Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn="arn:aws:iam::123456789012:role/mlf-sagemaker-job",
        stage_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-factory-stage:test",
    )
    repository = ExperimentRepository(settings=settings, store=store)
    orchestrator = InlineOrchestrator(store)
    experiments = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    uri = write_dataset(store, "churn.csv", golden_frame())
    record = experiments.create(name="golden", dataset_uri=uri, target_column=TARGET)
    experiments.save_features(record.experiment_id, selected_features=GOLDEN_FEATURES)
    experiments.start_training(record.experiment_id, TrainingRequest(models=GOLDEN_MODELS))
    if orchestrator.failures:
        raise RuntimeError(f"golden experiment failed: {orchestrator.failures}")
    return ExperimentLayout(base=record.artifact_prefix)


def experiment_id_of(layout: ExperimentLayout) -> str:
    return layout.base.rstrip("/").rsplit("/", 1)[-1]


def mask_json(text: str, experiment_id: str) -> str:
    """Canonical masked form of one artifact. Key order and key presence are preserved."""
    document = json.loads(text, object_pairs_hook=dict)
    return json.dumps(_mask(document, experiment_id), indent=2, ensure_ascii=False) + "\n"


def _mask(value: Any, experiment_id: str) -> Any:
    if isinstance(value, dict):
        masked = {}
        for key, item in value.items():
            if key in MASKED_KEYS and item is not None:
                masked[key] = MASK
            elif key in SORTED_KEYS and isinstance(item, list):
                masked[key] = sorted(_mask(item, experiment_id))
            else:
                masked[key] = _mask(item, experiment_id)
        return masked
    if isinstance(value, list):
        return [_mask(item, experiment_id) for item in value]
    if isinstance(value, str):
        return value.replace(experiment_id, EXPERIMENT_ID_MASK)
    return value


def collect_masked(store: ObjectStore, layout: ExperimentLayout) -> dict[str, str]:
    """Fixture file name -> masked artifact text, read from a finished experiment."""
    experiment_id = experiment_id_of(layout)
    return {
        name: mask_json(store.read_bytes(locate(layout)).decode("utf-8"), experiment_id)
        for name, locate in GOLDEN_FILES.items()
    }


def collect_raw(store: ObjectStore, layout: ExperimentLayout) -> dict[str, bytes]:
    """Fixture file name -> the artifact bytes exactly as the stage wrote them."""
    return {name: store.read_bytes(locate(layout)) for name, locate in GOLDEN_FILES.items()}


#: Valid stand-ins for masked values, so a fixture can be parsed with the contracts.
_PLACEHOLDERS: dict[str, Any] = {
    "duration_seconds": 0.0,
    "training_duration_seconds": 0.0,
    "etag": "etag",
    "size_bytes": 0,
    "version_id": "version",
    "package_versions": {},
    "platform": "platform",
    "python_version": "3",
}
_PLACEHOLDER_TIMESTAMP = "2026-01-01T00:00:00Z"


def golden_document(name: str) -> dict[str, Any]:
    """A golden fixture as a JSON document with masked values replaced by valid stand-ins."""
    return _unmask(json.loads(read_golden()[name]))


def _unmask(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (_PLACEHOLDERS.get(key, _PLACEHOLDER_TIMESTAMP) if item == MASK else _unmask(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_unmask(item) for item in value]
    return value


def read_golden() -> dict[str, str]:
    return {name: (GOLDEN_DIR / name).read_text(encoding="utf-8") for name in GOLDEN_FILES}


def write_golden(directory: Path = GOLDEN_DIR) -> list[Path]:
    from tests.support.in_memory_store import InMemoryObjectStore

    store = InMemoryObjectStore()
    layout = run_single_split_experiment(store)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in collect_masked(store, layout).items():
        path = directory / name
        path.write_bytes(text.encode("utf-8"))  # LF on every OS
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the golden fixtures")
    args = parser.parse_args(argv)
    if not args.write:
        parser.print_help()
        return 0
    for path in write_golden():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
