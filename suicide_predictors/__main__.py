import argparse
import logging
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from suicide_predictors import SCHEMA_VERSION
from suicide_predictors.artifacts import (
    PROJECT_ROOT,
    artifact_hashes,
    create_workspace,
    environment,
    file_hash,
    publish,
    source_hash,
    write_json,
)
from suicide_predictors.modeling import ExperimentConfig, train
from suicide_predictors.preparation import prepare


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Pipeline reproducible de análisis ENSANUT")
    result.add_argument("command", nargs="?", default="run", choices=("run", "prepare", "explore", "train", "explain"))
    result.add_argument("--raw", type=Path, default=PROJECT_ROOT / "data" / "adolescentes_ensanut2024_w.csv")
    result.add_argument("--workspace", type=Path)
    result.add_argument("--jobs", type=int, default=-1)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--min-features", type=int, default=3)
    result.add_argument("--outer-folds", type=int, default=5)
    result.add_argument("--inner-folds", type=int, default=3)
    result.add_argument("--final-folds", type=int, default=5)
    result.add_argument("--selection-trees", type=int, default=100)
    result.add_argument("--final-selection-trees", type=int, default=300)
    result.add_argument("--forest-trees", type=int, default=400)
    result.add_argument("--bootstrap", type=int, default=500)
    result.add_argument("--shap-samples", type=int, default=200)
    result.add_argument("--shap-background", type=int, default=50)
    result.add_argument("--skip-shap", action="store_true")
    return result


def experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        jobs=args.jobs,
        seed=args.seed,
        min_features=args.min_features,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        final_folds=args.final_folds,
        selection_trees=args.selection_trees,
        final_selection_trees=args.final_selection_trees,
        forest_trees=args.forest_trees,
        bootstrap_repetitions=args.bootstrap,
    )


def initial_manifest(args: argparse.Namespace, config: ExperimentConfig) -> dict:
    if args.command != "run":
        return {}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "raw_sha256": file_hash(args.raw),
        "source_sha256": source_hash(),
        "environment": environment(),
    }


def resolve_workspace(args: argparse.Namespace) -> Path:
    if args.workspace is not None:
        workspace = args.workspace.resolve()
    elif args.command == "run":
        workspace = create_workspace()
    else:
        workspace = PROJECT_ROOT / "output"
    if (workspace / "manifest.json").exists():
        raise ValueError("La ejecución ya existe. Usa un directorio nuevo para conservar los artefactos.")
    if args.command == "run":
        if workspace.exists() and any(workspace.iterdir()):
            raise ValueError("La ejecución necesita un directorio vacío")
    return workspace


def execute_stages(args: argparse.Namespace, workspace: Path, config: ExperimentConfig) -> None:
    if args.command in ("run", "prepare"):
        prepare(args.raw, workspace)
    if args.command in ("run", "explore"):
        from suicide_predictors.plots import explore

        descriptive_path = workspace / "preprocesamiento" / "adolescentes_limpio.csv"
        descriptive = pd.read_csv(descriptive_path)
        explore(descriptive, workspace)
    if args.command in ("run", "train"):
        ml_path = workspace / "preprocesamiento" / "adolescentes_ml.csv"
        frame = pd.read_csv(ml_path)
        train(frame, workspace, config)
    if args.command in ("run", "explain") and not args.skip_shap:
        from suicide_predictors.explain import explain

        explain(workspace, sample_size=args.shap_samples, background_size=args.shap_background)


def complete_run(args: argparse.Namespace, workspace: Path, manifest: dict) -> None:
    manifest.update(
        status="complete",
        completed_at=datetime.now(UTC).isoformat(),
        shap_enabled=not args.skip_shap,
        artifacts=artifact_hashes(workspace),
    )
    write_json(workspace / "manifest.json", manifest)
    if args.workspace is None:
        publish(workspace)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    config = experiment_config(args)
    manifest = initial_manifest(args, config)
    workspace = resolve_workspace(args)
    if args.command == "run":
        write_json(workspace / "manifest.json", manifest)
    try:
        execute_stages(args, workspace, config)
        if args.command == "run":
            complete_run(args, workspace, manifest)
        logging.info("Proceso completado: %s", workspace)
    except Exception as error:
        if args.command == "run":
            manifest.update(status="failed", error=str(error))
            write_json(workspace / "manifest.json", manifest)
        raise
    finally:
        if args.command == "run" and args.workspace is None and workspace.exists():
            shutil.rmtree(workspace)


if __name__ == "__main__":
    main()
