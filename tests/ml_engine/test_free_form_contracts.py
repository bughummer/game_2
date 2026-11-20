"""Free-form contracts + the FREE_FORM side workflow + layout tree (free-form design C1, C7)."""

from datetime import UTC, datetime

from ml_engine.contracts.execution import SIDE_WORKFLOWS, ComputeSpec, WorkflowKind
from ml_engine.contracts.free_form import (
    ApprovedScript,
    DataContextAnswer,
    DataContextAnswers,
    FreeFormMetric,
    FreeFormRequest,
    FreeFormResult,
    MetricLabel,
    ScriptRevision,
    content_hash,
)
from ml_engine.io import ExperimentLayout


def _approved(source: str = "print('hi')") -> ApprovedScript:
    return ApprovedScript(
        content_hash=content_hash(source),
        revision=1,
        predict_entry_point="predict",
        approved_by="owner",
        approved_at=datetime.now(UTC),
    )


def test_free_form_is_a_side_workflow():
    assert WorkflowKind.FREE_FORM in SIDE_WORKFLOWS


def test_content_hash_is_stable_and_distinct():
    assert content_hash("a") == content_hash("a")
    assert content_hash("a") != content_hash("b")


def test_contracts_round_trip():
    compute = ComputeSpec(instance_type="ml.m5.large", max_runtime_seconds=3600)
    models = [
        _approved(),
        ScriptRevision(
            revision=1,
            content="print('hi')",
            content_hash=content_hash("print('hi')"),
            authored_by="ai",
            created_at=datetime.now(UTC),
        ),
        DataContextAnswers(
            answers=[
                DataContextAnswer(
                    question_id="label_timing",
                    question="When was the label assigned?",
                    status="answered",
                    answer="at signup",
                )
            ]
        ),
        FreeFormRequest(
            run_id="ff-1",
            experiment_id="exp-1",
            approved_script=_approved(),
            dataset_uri="s3://b/data.csv",
            file_format="csv",
            compute=compute,
            requested_at=datetime.now(UTC),
        ),
        FreeFormResult(
            run_id="ff-1",
            experiment_id="exp-1",
            status="completed",
            generated_at=datetime.now(UTC),
            metrics=[FreeFormMetric(name="rmse", value=1.5, label=MetricLabel.ENGINE)],
            content_hash=content_hash("print('hi')"),
        ),
    ]
    for model in models:
        restored = type(model).model_validate_json(model.model_dump_json())
        assert restored == model


def test_result_detects_an_engine_metric():
    result = FreeFormResult(
        run_id="ff-1",
        experiment_id="exp-1",
        status="completed",
        generated_at=datetime.now(UTC),
        content_hash="x",
        metrics=[
            FreeFormMetric(name="acc", value=0.9, label=MetricLabel.SELF_REPORTED),
            FreeFormMetric(name="rmse", value=1.5, label=MetricLabel.ENGINE),
        ],
    )
    assert result.has_engine_metric


def test_layout_free_form_paths_are_per_run():
    layout = ExperimentLayout(base="s3://bucket/ml-factory/experiments/exp-1")
    a = layout.free_form_request("ff-a")
    b = layout.free_form_request("ff-b")
    assert a != b
    assert layout.free_form_prefix.endswith("free_form/")
    assert layout.free_form_script("ff-a").endswith("free_form/ff-a/script.py")
    assert layout.free_form_model_metadata("ff-a", "m").endswith(
        "free_form/ff-a/models/m/metadata.json"
    )
