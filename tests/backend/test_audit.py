"""The audit log (requirement 15.4): an append-only trail of mutations in the experiment prefix."""

from backend.audit import AuditEvent, AuditLog
from ml_engine.io import ExperimentLayout, read_model
from tests.support.datasets import ARTIFACT_ROOT


def test_an_event_is_written_under_the_experiment_prefix(store):
    layout = ExperimentLayout.for_experiment(ARTIFACT_ROOT, "exp-aud")
    audit = AuditLog(store)
    audit.record(
        artifact_prefix=layout.base,
        experiment_id="exp-aud",
        action="start_training",
        actor="dev@local",
        detail={"models": "logistic_regression"},
    )
    uris = store.list_uris(layout.path("audit") + "/")
    assert len(uris) == 1
    event = read_model(store, uris[0], AuditEvent)
    assert event.action == "start_training"
    assert event.actor == "dev@local"
    assert event.detail["models"] == "logistic_regression"


def test_each_event_is_its_own_object(store):
    layout = ExperimentLayout.for_experiment(ARTIFACT_ROOT, "exp-aud")
    audit = AuditLog(store)
    for action in ("create", "decide_features", "start_training"):
        audit.record(artifact_prefix=layout.base, experiment_id="exp-aud", action=action, actor="u")
    assert len(store.list_uris(layout.path("audit") + "/")) == 3  # single writer per object


def test_auditing_can_be_disabled(store):
    layout = ExperimentLayout.for_experiment(ARTIFACT_ROOT, "exp-aud")
    AuditLog(store, enabled=False).record(
        artifact_prefix=layout.base, experiment_id="exp-aud", action="x", actor="u"
    )
    assert store.list_uris(layout.path("audit") + "/") == []


def test_auditing_never_breaks_the_operation_it_records():
    class FailingStore:
        def write_bytes(self, *a, **k):
            raise RuntimeError("S3 down")

    # Must not raise, even though the write fails.
    AuditLog(FailingStore()).record(
        artifact_prefix=f"{ARTIFACT_ROOT}/ml-factory/experiments/exp-aud",
        experiment_id="exp-aud",
        action="x",
        actor="u",
    )
