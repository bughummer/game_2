"""The stage runtime as SageMaker runs it: the real ``build_store`` over (moto) S3.

Every other job test substitutes the in-memory store. This one substitutes nothing but the AWS
endpoint, so it proves the production wiring: stage CLI → ``build_store`` → ``S3ObjectStore``.
"""

import boto3
import pytest
from moto import mock_aws

from jobs._common.runtime import build_store
from jobs.profiling.main import main as profiling_main
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.io import ExperimentLayout, S3ObjectStore, read_model

BUCKET = "mlf-runtime-test"


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def test_build_store_is_always_s3(s3):
    assert isinstance(build_store(), S3ObjectStore)


def test_build_store_applies_kms_settings_from_the_job_environment(s3, monkeypatch):
    monkeypatch.setenv("ML_FACTORY_S3_SSE", "aws:kms")
    monkeypatch.setenv("ML_FACTORY_S3_KMS_KEY_ID", "alias/mlf")
    store = build_store()
    assert store._write_args == {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "alias/mlf"}


def test_profiling_stage_runs_end_to_end_against_s3(s3, classification_frame):
    dataset_uri = f"s3://{BUCKET}/curated/churn.csv"
    s3.put_object(
        Bucket=BUCKET, Key="curated/churn.csv", Body=classification_frame.to_csv(index=False)
    )
    layout = ExperimentLayout.for_experiment(f"s3://{BUCKET}", "exp-s3")

    code = profiling_main(
        [
            "--experiment-id",
            "exp-s3",
            "--artifact-base",
            layout.base,
            "--dataset-uri",
            dataset_uri,
            "--target-column",
            "churned",
        ]
    )

    assert code == 0
    store = S3ObjectStore(s3)
    eda = read_model(store, layout.eda, EdaReport)
    leakage = read_model(store, layout.leakage, LeakageReport)
    assert eda.dataset.row_count == len(classification_frame)
    assert leakage.findings, "the planted churn_copy leak must be reported"
    key = layout.eda.removeprefix(f"s3://{BUCKET}/")
    assert s3.head_object(Bucket=BUCKET, Key=key)["ServerSideEncryption"] == "AES256"


def test_a_stage_failure_lands_in_s3_as_an_artifact(s3, classification_frame):
    s3.put_object(
        Bucket=BUCKET, Key="curated/churn.csv", Body=classification_frame.to_csv(index=False)
    )
    layout = ExperimentLayout.for_experiment(f"s3://{BUCKET}", "exp-bad")

    code = profiling_main(
        [
            "--experiment-id",
            "exp-bad",
            "--artifact-base",
            layout.base,
            "--dataset-uri",
            f"s3://{BUCKET}/curated/churn.csv",
            "--target-column",
            "does_not_exist",
        ]
    )

    assert code == 1
    assert S3ObjectStore(s3).exists(f"{layout.base}/eda/failure.json")
