"""Boundary checks for the agent contracts and layout (ml-assistant task 1.6).

Requirements 15.1, 15.3, 15.5, 16.6: the agent contracts import only ``pydantic`` and the
standard library, agent storage paths are built only in ``ml_engine/io/layout.py``, and the
inherited manifest check still passes.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from ml_engine.io.layout import AGENT_NAMESPACE, AgentLayout, owner_key

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "ml_engine" / "contracts" / "agent.py"
LAYOUT = ROOT / "ml_engine" / "io" / "layout.py"
SOURCE_PACKAGES = ("backend", "ml_engine", "mcp_server", "jobs", "container", "scripts")

# Literals that only make sense when building an agent storage path.
AGENT_PATH_MARKERS = ("ml-factory/agent", "sessions/", "/inflight/", "/owners/")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_modules(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                found.add(".")  # any relative import leaves the stdlib + pydantic set
            elif node.module:
                found.add(node.module.split(".")[0])
    return found


def _source_files() -> list[Path]:
    files: list[Path] = []
    for package in SOURCE_PACKAGES:
        base = ROOT / package
        if base.is_dir():
            files.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


def test_agent_contracts_import_only_pydantic_and_the_standard_library():
    imported = _top_level_modules(_parse(CONTRACTS))
    allowed = set(sys.stdlib_module_names) | {"pydantic", "__future__"}
    assert imported - allowed == set(), f"forbidden imports: {sorted(imported - allowed)}"
    assert "pydantic" in imported


def test_agent_contracts_build_no_storage_paths():
    literals = [
        node.value
        for node in ast.walk(_parse(CONTRACTS))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    # Docstrings may mention layout.py; no literal may hold an agent path fragment.
    offending = [s for s in literals if any(m in s for m in ("ml-factory/agent", "/inflight/"))]
    assert not offending


def test_agent_paths_are_built_only_in_layout():
    offenders: list[str] = []
    for path in _source_files():
        if path == LAYOUT:
            continue
        tree = _parse(path)
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and any(marker in node.value for marker in AGENT_PATH_MARKERS)
            ):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    # scripts/inherited_manifest.py describes layout.py in prose, not as a path.
    offenders = [o for o in offenders if not o.startswith("scripts/inherited_manifest.py:")]
    assert not offenders, f"agent paths built outside ml_engine/io/layout.py: {offenders}"


def test_agent_namespace_constant_is_used_only_in_layout():
    users = [
        path.relative_to(ROOT).as_posix()
        for path in _source_files()
        if path != LAYOUT and "AGENT_NAMESPACE" in path.read_text(encoding="utf-8")
    ]
    users = [u for u in users if u != "scripts/inherited_manifest.py"]
    assert not users, f"AGENT_NAMESPACE used outside layout.py: {users}"


def test_agent_layout_paths_live_under_the_agent_namespace():
    root = "s3://bucket/prefix"
    layout = AgentLayout.for_session(root, "sess-1")
    base = f"{root}/{AGENT_NAMESPACE}/sessions/sess-1"
    assert layout.base == base
    assert layout.session == f"{base}/session.json"
    assert layout.tool_result("tc-1") == f"{base}/results/tc-1.json"
    assert layout.inflight("k1") == f"{base}/inflight/k1.json"
    assert layout.audit_prefix == f"{base}/audit/"
    key = owner_key("alice@example.com")
    assert AgentLayout.owner_index_prefix(root, "alice@example.com") == (
        f"{root}/{AGENT_NAMESPACE}/owners/{key}/"
    )
    assert AgentLayout.owner_index_object(root, "alice@example.com", "sess-1") == (
        f"{root}/{AGENT_NAMESPACE}/owners/{key}/sess-1.json"
    )


def test_owner_key_is_stable_and_path_safe():
    key = owner_key("bob/../x y")
    assert key == owner_key("bob/../x y")
    assert len(key) == 24 and all(c in "0123456789abcdef" for c in key)
    assert owner_key("a") != owner_key("b")


def test_inherited_manifest_check_still_passes():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(ROOT / "tests" / "test_inherited_manifest.py"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
