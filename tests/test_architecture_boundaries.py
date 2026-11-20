"""The dependency rules from .kiro/steering/architecture.md, enforced (design Property 1).

If one of these fails, fix the import — do not relax the rule.

Adapted from old_repo: the workflow/queue rules were dropped (there is no worker); the AWS rule
was tightened (``ml_engine/io/s3.py`` takes an injected client, so no module in ``ml_engine``
imports the SDK at all); SageMaker SDK and MCP imports were added to the forbidden set. Rules for
``mcp_server`` are added when that package lands.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _name(module: Path) -> str:
    """Package-relative module name with forward slashes, on every platform."""
    return module.relative_to(ROOT).as_posix()


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(nodes) -> set[str]:
    names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _all_imports(path: Path) -> set[str]:
    return _imports(ast.walk(_parse(path)))


def _module_level_imports(path: Path) -> set[str]:
    """Imports executed when the module is imported (top level, incl. try/if blocks)."""
    found: set[str] = set()
    pending = list(_parse(path).body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        found |= _imports([node])
        for field in ("body", "orelse", "finalbody", "handlers"):
            pending.extend(getattr(node, field, []) or [])
    return found


def _modules(package: str) -> list[Path]:
    return sorted(p for p in (ROOT / package).rglob("*.py") if "__pycache__" not in p.parts)


def _roots(names: set[str]) -> set[str]:
    return {n.split(".")[0] for n in names}


@pytest.mark.parametrize("module", _modules("ml_engine"), ids=_name)
def test_ml_engine_does_not_depend_on_the_platform(module: Path) -> None:
    forbidden = {
        "fastapi",
        "starlette",
        "pydantic_settings",
        "backend",
        "jobs",
        "mcp_server",
        "mcp",
        "fastmcp",
    }
    offenders = _roots(_all_imports(module)) & forbidden
    assert not offenders, f"{_name(module)} imports {offenders}"


@pytest.mark.parametrize("module", _modules("ml_engine"), ids=_name)
def test_ml_engine_never_imports_an_aws_sdk(module: Path) -> None:
    """AWS clients are injected; the deterministic core stays installable without the SDK."""
    offenders = _roots(_all_imports(module)) & {"boto3", "botocore", "sagemaker", "aiobotocore"}
    assert not offenders, f"{_name(module)} imports {offenders}; inject the client instead"


@pytest.mark.parametrize("module", _modules("jobs"), ids=_name)
def test_jobs_do_not_depend_on_the_control_plane(module: Path) -> None:
    forbidden = {"backend", "fastapi", "starlette", "mcp_server", "mcp", "fastmcp", "sagemaker"}
    offenders = _roots(_all_imports(module)) & forbidden
    assert not offenders, f"{_name(module)} imports {offenders}"


@pytest.mark.parametrize("module", _modules("jobs"), ids=_name)
def test_jobs_import_boto3_only_lazily_in_the_runtime(module: Path) -> None:
    """Stages stay importable without AWS; only ``build_store`` reaches for boto3, at call time."""
    aws = {"boto3", "botocore"}
    assert not (_roots(_module_level_imports(module)) & aws), (
        f"{_name(module)} imports the AWS SDK at module import time"
    )
    if _name(module) != "jobs/_common/runtime.py":
        assert not (_roots(_all_imports(module)) & aws), (
            f"{_name(module)} uses the AWS SDK; only jobs/_common/runtime.py::build_store may"
        )


def test_no_local_store_or_file_uris_survive() -> None:
    """There is no local execution path (design DD-6). Nothing may bring one back quietly."""
    offenders = []
    for package in ("ml_engine", "jobs", "backend"):
        for module in _modules(package):
            text = module.read_text(encoding="utf-8")
            code = "\n".join(
                line for line in text.splitlines() if not line.lstrip().startswith("#")
            )
            for needle in ("LocalObjectStore(", "to_file_uri(", "file_uri_to_path(", "'file://"):
                if needle in code or needle.replace("'", '"') in code:
                    offenders.append(f"{_name(module)}: {needle}")
    assert not offenders, f"local-storage code found: {offenders}"


#: The one control-plane module allowed to reach for the ``sagemaker`` SDK — lazily, inside a
#: function — to resolve the stock stage image (stock-image migration, design C1, Req 1.2/6.2).
#: ``ml_engine`` and ``jobs`` still never import it (the two tests above enforce that).
SAGEMAKER_SDK_READERS = {"backend/aws/images.py"}


@pytest.mark.parametrize("module", _modules("backend"), ids=_name)
def test_the_control_plane_never_runs_a_stage(module: Path) -> None:
    """The control plane submits work to SageMaker; it never imports a stage to run it.

    ``backend/aws/images.py`` may use the ``sagemaker`` SDK (lazily) to resolve the stock image;
    that is the sanctioned AWS adapter layer, like ``backend/aws/clients.py`` for boto3.
    """
    forbidden = {"jobs", "mcp_server", "fastmcp"}
    if _name(module) not in SAGEMAKER_SDK_READERS:
        forbidden = forbidden | {"sagemaker"}
    offenders = _roots(_all_imports(module)) & forbidden
    assert not offenders, f"{_name(module)} imports {offenders}; stages run on SageMaker"


@pytest.mark.parametrize("module", _modules("backend"), ids=_name)
def test_the_sagemaker_sdk_is_imported_only_lazily_in_the_image_resolver(module: Path) -> None:
    """Even the sanctioned reader imports the SDK inside a function, never at module load, so the
    control plane stays importable without the SDK (stock-image migration, design C1)."""
    assert "sagemaker" not in _roots(_module_level_imports(module)), (
        f"{_name(module)} imports the sagemaker SDK at module import time; import it lazily"
    )


@pytest.mark.parametrize("module", _modules("backend"), ids=_name)
def test_the_control_plane_imports_boto3_only_lazily(module: Path) -> None:
    """Clients are built in backend/aws and injected; modules stay importable without AWS."""
    aws = {"boto3", "botocore"}
    assert not (_roots(_module_level_imports(module)) & aws), (
        f"{_name(module)} imports the AWS SDK at module import time"
    )


def test_one_list_of_resumable_models() -> None:
    """Spot checkpoint/resume is dropped (stock-image migration, decision B4): no model resumes,
    and the training stage shares the one (now empty) RESUMABLE_MODELS constant."""
    from jobs.training import checkpointing
    from ml_engine.contracts.execution import RESUMABLE_MODELS

    assert checkpointing.RESUMABLE_MODELS is RESUMABLE_MODELS
    assert frozenset() == RESUMABLE_MODELS


#: The only modules allowed to read the environment. Everything else takes configuration as an
#: argument, which is what makes a deployment auditable from one file per process.
ENVIRONMENT_READERS = {
    "jobs/_common/runtime.py",  # log level and S3 encryption settings for a stage
    "backend/config.py",  # pydantic-settings; the control plane's one configuration module
}


@pytest.mark.parametrize(
    "module", _modules("jobs") + _modules("ml_engine") + _modules("backend"), ids=_name
)
def test_configuration_is_read_in_one_place_per_process(module: Path) -> None:
    """A setting read straight from the environment is a setting nobody can find later."""
    if _name(module) in ENVIRONMENT_READERS:
        return
    text = module.read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in text.splitlines()
        if ("os.environ" in line or "os.getenv" in line) and not line.strip().startswith("#")
    ]
    assert not offenders, f"{_name(module)} reads the environment directly: {offenders}"


def test_artifact_paths_are_defined_only_in_the_layout() -> None:
    """No module may hardcode an artifact path; ExperimentLayout owns them."""
    offenders = []
    for package in ("ml_engine", "jobs", "backend"):
        for module in _modules(package):
            if module.name == "layout.py":
                continue
            text = module.read_text(encoding="utf-8")
            for needle in (
                '"eda/eda.json"',
                '"comparison/comparison.json"',
                '"models/"',
                # cross-validation and Test_Fold artifacts
                '"validation/fold_assignment.json"',
                '"datasets/development.parquet"',
                '"datasets/test.parquet"',
                '"comparison/test_evaluation.json"',
                '"preprocessing/folds/',
            ):
                if needle in text:
                    offenders.append(f"{_name(module)}: {needle}")
    assert not offenders, f"hardcoded artifact paths: {offenders}"
