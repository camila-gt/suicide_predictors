import hashlib
import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def active_workspace(root: Path = PROJECT_ROOT) -> Path:
    pointer = root / "output" / "current.json"
    if not pointer.exists():
        return root
    current = json.loads(pointer.read_text(encoding="utf-8"))
    workspace = (root / current["workspace"]).resolve()
    runs = (root / "output" / "runs").resolve()
    if not workspace.is_relative_to(runs):
        raise ValueError("La ejecución activa está fuera del directorio de artefactos")
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("La ejecución activa no está completa")
    return workspace


def create_workspace(root: Path = PROJECT_ROOT) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    workspace = root / "output" / "runs" / run_id
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def publish(workspace: Path, root: Path = PROJECT_ROOT) -> None:
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Solo se puede publicar una ejecución completa")
    write_json(root / "output" / "current.json", {"workspace": workspace.relative_to(root).as_posix()})


def environment() -> dict[str, str]:
    import platform

    packages = ("numpy", "pandas", "scikit-learn", "scipy", "shap", "matplotlib", "joblib", "streamlit", "plotly")
    versions = {"python": platform.python_version()}
    for package in packages:
        versions[package] = version(package)
    return versions


def artifact_hashes(workspace: Path) -> dict[str, str]:
    hashes = {}
    for directory in ("data", "output", "figuras"):
        for path in (workspace / directory).rglob("*"):
            if path.is_file():
                relative_path = path.relative_to(workspace).as_posix()
                hashes[relative_path] = file_hash(path)
    return hashes


def source_hash(root: Path = PROJECT_ROOT) -> str:
    files = []
    for directory in ("suicide_predictors", "dashboard"):
        files.extend((root / directory).rglob("*.py"))
    files.sort()
    files += [root / "requirements.txt", root / "run_pipeline.ps1", root / "pyproject.toml"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
