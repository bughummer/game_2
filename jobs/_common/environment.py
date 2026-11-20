"""Capture the job's own runtime environment and write it next to the experiment's artifacts."""

import platform
import re
from datetime import UTC, datetime
from importlib import metadata

from jobs._common.runtime import stage_image_uri
from ml_engine.contracts.environment import RuntimeEnvironment
from ml_engine.io import ExperimentLayout, ObjectStore, write_model

#: The platform's own package: installed from the image's source tree, not from an index.
OWN_PACKAGE = "ml-factory"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def capture(experiment_id: str, stage: str) -> RuntimeEnvironment:
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[_normalize(name)] = distribution.version
    return RuntimeEnvironment(
        experiment_id=experiment_id,
        stage=stage,
        captured_at=datetime.now(UTC),
        image_uri=stage_image_uri(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        packages=dict(sorted(packages.items())),
    )


def requirements_txt(environment: RuntimeEnvironment) -> str:
    """Exact pins for every installed package, pip format, with the context pip cannot express."""
    lines = [
        "# Environment the models of this experiment were trained in (captured inside the job).",
        f"# python {environment.python_version}  ({environment.platform})",
        f"# image {environment.image_uri or 'unknown'}",
        f"# captured {environment.captured_at.isoformat()}",
        f"# {OWN_PACKAGE} itself is installed from the image source, not from an index.",
    ]
    for name, version in environment.packages.items():
        line = f"{name}=={version}"
        lines.append(f"# {line}" if name == OWN_PACKAGE else line)
    return "\n".join(lines) + "\n"


def write_training_environment(
    store: ObjectStore, layout: ExperimentLayout, experiment_id: str
) -> RuntimeEnvironment:
    """Record the training environment. Every model job of a run writes identical content."""
    environment = capture(experiment_id, "training")
    write_model(store, layout.environment, environment)
    store.write_bytes(layout.requirements, requirements_txt(environment).encode("utf-8"))
    return environment
