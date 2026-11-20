"""Resolve the SageMaker stage image URI for a run (stock-image migration, design C1).

Every Processing and Training job runs one AWS-prebuilt image — the SageMaker XGBoost image, which
also ships scikit-learn. The control plane resolves the image URI itself at submission: the
configured override if set, else an AWS SDK lookup for the region. No image is built or pushed by
us, and no URI is typed by a person in the common case.

The ``sagemaker`` SDK is imported lazily inside the lookup so ``ml_engine`` keeps no AWS dependency
and the default test suite needs no SDK unless it exercises the lookup path.
"""

from dataclasses import dataclass
from typing import Literal

from backend.config import Settings

#: The AWS prebuilt framework whose image also ships scikit-learn, pandas, numpy, scipy.
STAGE_IMAGE_FRAMEWORK = "xgboost"


class ImageResolutionError(RuntimeError):
    """The stage image could not be resolved; no job may be submitted.

    ``code`` is stable and user-safe; ``region`` is the region the lookup was attempted for.
    """

    def __init__(self, code: str, message: str, region: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.region = region


@dataclass(frozen=True, slots=True)
class ResolvedImage:
    """The image a run will use, and where the URI came from."""

    uri: str
    source: Literal["config", "aws_lookup"]


def resolve_stage_image(settings: Settings) -> ResolvedImage:
    """Return the stage image URI for a run.

    - ``settings.stage_image_uri`` set -> that URI, ``source="config"`` (an explicit override, e.g.
      for pinning a specific digest).
    - unset -> the AWS SageMaker XGBoost image for ``settings.aws_region`` via
      ``sagemaker.image_uris.retrieve``, ``source="aws_lookup"``.

    Raises ``ImageResolutionError`` (naming the region) when the SDK is absent or the lookup fails,
    so a missing dependency or an unknown region fails before anything is submitted.
    """
    configured = (settings.stage_image_uri or "").strip()
    if configured:
        return ResolvedImage(uri=configured, source="config")

    try:
        from sagemaker import image_uris
    except ImportError as error:  # the SDK is a control-plane dependency, not an engine one
        raise ImageResolutionError(
            "SAGEMAKER_SDK_MISSING",
            "The sagemaker SDK is not installed, so the stage image cannot be resolved. "
            "Install the 'backend' extra or set ML_FACTORY_STAGE_IMAGE_URI explicitly.",
            settings.aws_region,
        ) from error

    try:
        uri = image_uris.retrieve(
            framework=STAGE_IMAGE_FRAMEWORK,
            region=settings.aws_region,
            version=settings.xgboost_image_version,
        )
    except Exception as error:  # unknown region, unknown version, SDK-internal failure
        raise ImageResolutionError(
            "IMAGE_LOOKUP_FAILED",
            f"Could not resolve the AWS {STAGE_IMAGE_FRAMEWORK} image for region "
            f"{settings.aws_region!r}: {error}. Set ML_FACTORY_STAGE_IMAGE_URI to override.",
            settings.aws_region,
        ) from error

    if not uri:
        raise ImageResolutionError(
            "IMAGE_LOOKUP_EMPTY",
            f"The AWS {STAGE_IMAGE_FRAMEWORK} image lookup returned nothing for region "
            f"{settings.aws_region!r}.",
            settings.aws_region,
        )
    return ResolvedImage(uri=uri, source="aws_lookup")
