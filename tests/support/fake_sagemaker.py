"""A SageMaker client double that validates every request against the real API model.

Request parameters are checked with botocore's own ``ParamValidator`` against the installed
SageMaker service model, so a malformed ``create_training_job`` fails the test exactly as AWS
would reject it — without network or an account. Responses are scripted per job name.
"""

from typing import Any

import botocore.session
from botocore.validate import ParamValidator

_MODEL = botocore.session.get_session().get_service_model("sagemaker")
_VALIDATOR = ParamValidator()

OPERATIONS = {
    "create_training_job": "CreateTrainingJob",
    "create_processing_job": "CreateProcessingJob",
    "describe_training_job": "DescribeTrainingJob",
    "describe_processing_job": "DescribeProcessingJob",
    "stop_training_job": "StopTrainingJob",
    "stop_processing_job": "StopProcessingJob",
    "list_training_jobs": "ListTrainingJobs",
}
#: Operations the platform must never call (requirement 4.6: no inference endpoints).
FORBIDDEN_PREFIXES = ("create_endpoint", "create_model", "create_transform_job")


class ClientError(Exception):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}")
        self.response = {"Error": {"Code": code, "Message": message}}


class FakeSageMaker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.created: dict[str, dict[str, Any]] = {}
        self.status: dict[str, dict[str, Any]] = {}
        self.fail_create: dict[str, Exception] = {}
        self.fail_stop: dict[str, Exception] = {}
        #: In-progress training jobs owned by other experiments (platform-wide cap tests).
        self.external_training_jobs: list[dict[str, Any]] = []

    def __getattr__(self, name: str):
        if name.startswith(FORBIDDEN_PREFIXES):
            raise AssertionError(f"the platform must never call {name}")
        if name not in OPERATIONS:
            raise AttributeError(name)
        return lambda **kwargs: self._call(name, kwargs)

    # --- scripting ---------------------------------------------------------------------------
    def set_status(self, job_name: str, status: str, **extra: Any) -> None:
        """Script a describe response. ``extra`` may include SecondaryStatus, FailureReason, etc."""
        self.status[job_name] = {"status": status, **extra}

    def names(self, operation: str) -> list[str]:
        key = "TrainingJobName" if "training" in operation else "ProcessingJobName"
        return [kwargs[key] for op, kwargs in self.calls if op == operation]

    # --- dispatch ----------------------------------------------------------------------------
    def _call(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        shape = _MODEL.operation_model(OPERATIONS[name]).input_shape
        report = _VALIDATOR.validate(kwargs, shape)
        if report.has_errors():
            raise AssertionError(
                f"{name} request rejected by the API model:\n{report.generate_report()}"
            )
        self.calls.append((name, kwargs))
        if name.startswith("create_"):
            job = kwargs.get("TrainingJobName") or kwargs["ProcessingJobName"]
            if job in self.fail_create:
                raise self.fail_create[job]
            if job in self.created:
                raise ClientError("ResourceInUse", f"{job} already exists")
            self.created[job] = kwargs
            self.status.setdefault(job, {"status": "InProgress"})
            kind = "Training" if "training" in name else "Processing"
            return {f"{kind}JobArn": f"arn:aws:sagemaker:us-east-1:123456789012:job/{job}"}
        if name.startswith("describe_"):
            job = kwargs.get("TrainingJobName") or kwargs["ProcessingJobName"]
            if job not in self.created:
                raise ClientError("ValidationException", f"{job} not found")
            scripted = dict(self.status[job])
            status = scripted.pop("status")
            key = "TrainingJobStatus" if "training" in name else "ProcessingJobStatus"
            return {key: status, **scripted}
        if name == "list_training_jobs":
            wanted = kwargs.get("StatusEquals")
            contains = kwargs.get("NameContains", "")
            summaries = [
                {"TrainingJobName": job, "TrainingJobStatus": self.status[job]["status"]}
                for job, request in self.created.items()
                if "TrainingJobName" in request
                and contains in job
                and (wanted is None or self.status[job]["status"] == wanted)
            ]
            return {"TrainingJobSummaries": summaries + list(self.external_training_jobs)}
        if name.startswith("stop_"):
            job = kwargs.get("TrainingJobName") or kwargs["ProcessingJobName"]
            if job in self.fail_stop:
                raise self.fail_stop[job]
        return {}


class FakeLogs:
    """A minimal CloudWatch Logs double: scripted events, or an error that must be swallowed."""

    def __init__(self) -> None:
        self.events: dict[str, list[str]] = {}
        self.raise_error = False

    def set_events(self, job_name: str, lines: list[str]) -> None:
        self.events[job_name] = lines

    def filter_log_events(self, logGroupName, logStreamNamePrefix, limit):  # noqa: N803
        if self.raise_error:
            raise RuntimeError("ResourceNotFoundException: log group does not exist")
        lines = self.events.get(logStreamNamePrefix, [])[:limit]
        return {"events": [{"message": line} for line in lines]}
