"""The runtime environment a stage actually ran in (ML Factory 2.0).

Captured inside the job container — not in the control plane — so it describes the image that
built (or scored) the models: image digest, Python, platform, and every installed package with
its exact version. ``environment/requirements.txt`` is the same package list in pip format.
"""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import StrictModel

ENVIRONMENT_SCHEMA_VERSION = "1.0"

#: Packages whose version changes can change model results or break loading a saved model.
RESULT_CRITICAL_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "xgboost",
    "catboost",
    "pyarrow",
)


class RuntimeEnvironment(StrictModel):
    schema_version: str = ENVIRONMENT_SCHEMA_VERSION
    experiment_id: str
    stage: str
    captured_at: datetime
    image_uri: str | None = Field(
        default=None, description="The stage image (digest) the job ran on, when known."
    )
    python_version: str
    platform: str
    packages: dict[str, str] = Field(
        default_factory=dict, description="Every installed distribution: name -> version."
    )

    def differences(self, other: "RuntimeEnvironment") -> dict[str, str]:
        """Packages whose version differs (or is missing on one side): name -> 'this -> other'."""
        names = set(self.packages) | set(other.packages)
        changed = {
            name: f"{self.packages.get(name, 'absent')} -> {other.packages.get(name, 'absent')}"
            for name in sorted(names)
            if self.packages.get(name) != other.packages.get(name)
        }
        if self.python_version != other.python_version:
            changed["python"] = f"{self.python_version} -> {other.python_version}"
        return changed
