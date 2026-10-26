import hashlib
import json
import os
import shutil
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
    return root / "output"


def create_workspace(root: Path = PROJECT_ROOT) -> Path:
    workspace = root / ".pipeline-work"
    try:
        workspace.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(
            "Ya existe un temporal .pipeline-work; comprueba que no haya otro pipeline ejecutándose"
        ) from error
    return workspace


def publish(workspace: Path, root: Path = PROJECT_ROOT) -> None:
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Solo se puede publicar una ejecución completa")
    output = active_workspace(root)
    backup = root / ".output-backup"
    if workspace.resolve() == output.resolve():
        raise ValueError("La publicación requiere un directorio temporal independiente")
    if backup.exists():
        raise ValueError("Existe un respaldo de publicación pendiente: .output-backup")
    had_output = output.exists()
    if had_output:
        output.rename(backup)
    try:
        workspace.rename(output)
    except Exception:
        if had_output:
            backup.rename(output)
        raise
    if had_output:
        shutil.rmtree(backup)


def environment() -> dict[str, str]:
    import platform

    packages = ("numpy", "pandas", "scikit-learn", "scipy", "shap", "matplotlib", "joblib", "streamlit", "plotly")
    versions = {"python": platform.python_version()}
    for package in packages:
        versions[package] = version(package)
    return versions


def artifact_hashes(workspace: Path) -> dict[str, str]:
    return {
        path.relative_to(workspace).as_posix(): file_hash(path)
        for path in sorted(workspace.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


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
