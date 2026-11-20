"""Per-experiment package requirements (stock-image migration, decision B10/B11/B13; Stage 2).

An experiment may carry a user-authored list of pip requirements — its own ``requirements.txt`` —
installed on top of the stock AWS image at job start so the model catalogue is open to anything on
PyPI (CatBoost, LightGBM, …) with no custom image. This module holds the versioned contract and a
pure well-formedness check; the install step and the resolved ``pip freeze`` are Stage 2 job work.

Validation here is deliberately *non-restricting* (decision B13, trusted users): it rejects only
malformed specifiers (typos caught before submission), not choices. An optional allowlist is
enforced by the service layer, off by default.
"""

from pydantic import Field, field_validator

from ml_engine.contracts.common import StrictModel

REQUIREMENTS_SCHEMA_VERSION = "1.0"


class RequirementsError(ValueError):
    """A requirements entry is malformed. Raised before submission so no job is started."""


def validate_requirement_line(line: str) -> str:
    """Return the normalised requirement string, or raise ``RequirementsError``.

    A well-formed PEP 508 requirement (``name``, ``name>=1.2``, ``name[extra]==1.0``). Blank lines
    and ``#`` comments are rejected here — the caller strips them first — because an entry in the
    list is meant to be an actual package. Uses ``packaging`` when available (it ships with pip),
    with a conservative regex fallback so the engine needs no hard dependency.
    """
    text = (line or "").strip()
    if not text or text.startswith("#"):
        raise RequirementsError(f"not a package requirement: {line!r}")
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:  # pragma: no cover - packaging is present in every supported environment
        return _validate_with_regex(text)
    try:
        return str(Requirement(text))
    except InvalidRequirement as error:
        raise RequirementsError(f"malformed requirement {line!r}: {error}") from error


def requirement_name(line: str) -> str:
    """The distribution name of a requirement string (``catboost>=1.2`` -> ``catboost``).

    Lower-cased and PEP 503-normalised (``_``/``.`` -> ``-``) so an allowlist match is robust.
    """
    text = (line or "").strip()
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name

        return canonicalize_name(Requirement(text).name)
    except Exception:  # pragma: no cover - fallback when packaging is unavailable or text is odd
        import re

        match = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", text)
        name = match.group(0) if match else text
        return name.lower().replace("_", "-").replace(".", "-")


def _validate_with_regex(text: str) -> str:
    import re

    # name (PEP 503) optionally followed by [extras] and/or a version spec; no URLs or markers in
    # the fallback path.
    pattern = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._-]*"  # distribution name
        r"(\[[A-Za-z0-9,._-]+\])?"  # optional extras
        r"\s*((==|>=|<=|~=|!=|>|<)\s*[A-Za-z0-9][A-Za-z0-9.*+!._-]*"  # optional version op
        r"(\s*,\s*(==|>=|<=|~=|!=|>|<)\s*[A-Za-z0-9][A-Za-z0-9.*+!._-]*)*)?$"
    )
    if not pattern.match(text):
        raise RequirementsError(f"malformed requirement {text!r}")
    return text


class RequirementsSpec(StrictModel):
    """The approved per-experiment pip requirements, frozen into the experiment configuration.

    Empty ``packages`` reproduces today's behaviour (nothing installed). Each entry is validated as
    a PEP 508 requirement; duplicates are collapsed, order preserved.
    """

    schema_version: str = REQUIREMENTS_SCHEMA_VERSION
    packages: list[str] = Field(
        default_factory=list,
        description="User-approved pip requirements installed on the stock image before training.",
    )

    @field_validator("packages")
    @classmethod
    def _well_formed(cls, value: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for entry in value:
            normalised = validate_requirement_line(entry)
            seen.setdefault(normalised, None)
        return list(seen)

    @property
    def is_empty(self) -> bool:
        return not self.packages
