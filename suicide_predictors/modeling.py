import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn import config_context
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from suicide_predictors import SCHEMA_VERSION
from suicide_predictors.artifacts import write_json
from suicide_predictors.estimators import SurveyClassifier, prediction_pipeline
from suicide_predictors.evaluation import calibration_table, group_bootstrap, metrics
from suicide_predictors.schema import DESIGN_COLUMNS, FEATURE_CHANNELS, FEATURES, TARGET

LOGGER = logging.getLogger(__name__)
MODEL_NAMES = ("Regresión Logística", "Random Forest", "Gradient Boosting", "SVM (RBF)")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    final_folds: int = 5
    selection_trees: int = 100
    final_selection_trees: int = 300
    forest_trees: int = 400
    min_features: int = 3
    jobs: int = -1
    bootstrap_repetitions: int = 500

    def __post_init__(self) -> None:
        fold_counts = (self.outer_folds, self.inner_folds, self.final_folds)
        if min(fold_counts) < 2:
            raise ValueError("La validación requiere al menos dos pliegues")
        positive_counts = (self.selection_trees, self.final_selection_trees, self.forest_trees, self.min_features)
        if min(positive_counts) < 1:
            raise ValueError("El número de árboles y predictores debe ser positivo")
        if self.bootstrap_repetitions < 0 or self.jobs == 0:
            raise ValueError("Configuración inválida de bootstrap o paralelismo")


@dataclass(frozen=True)
class ModelData:
    features: pd.DataFrame
    target: pd.Series
    groups: pd.Series
    weights: pd.Series
    row_ids: pd.Series

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "ModelData":
        return cls(
            features=frame[list(FEATURES)],
            target=frame[TARGET],
            groups=frame["upm"],
            weights=frame["ponde_f"],
            row_ids=frame["row_id"],
        )

    def take(self, positions: np.ndarray) -> "ModelData":
        return ModelData(
            features=self.features.iloc[positions],
            target=self.target.iloc[positions],
            groups=self.groups.iloc[positions],
            weights=self.weights.iloc[positions],
            row_ids=self.row_ids.iloc[positions],
        )


@dataclass
class CrossValidationResult:
    scores: pd.DataFrame
    predictions: pd.DataFrame
    selections: list[dict]
    fold_assignments: np.ndarray


@dataclass
class TestEvaluation:
    model: Pipeline
    predictions: np.ndarray
    probabilities: np.ndarray


def validate_ml(frame: pd.DataFrame) -> None:
    required_columns = {*FEATURES, *DESIGN_COLUMNS, TARGET}
    actual_columns = set(frame.columns)
    if actual_columns != required_columns:
        missing = required_columns - actual_columns
        unexpected = actual_columns - required_columns
        raise ValueError(f"Esquema ML incompatible: faltan {missing}, sobran {unexpected}")
    identifiers_missing = frame[list(DESIGN_COLUMNS)].isna().any().any()
    duplicate_rows = frame["row_id"].duplicated().any()
    if identifiers_missing or duplicate_rows:
        raise ValueError("Identificadores ausentes o filas duplicadas")
    if set(frame[TARGET]) != {0, 1}:
        raise ValueError("El desenlace debe contener únicamente ambas clases binarias")
    weights_are_finite = np.isfinite(frame["ponde_f"]).all()
    weights_are_positive = frame["ponde_f"].gt(0).all()
    if not weights_are_finite or not weights_are_positive:
        raise ValueError("Ponderadores inválidos")
    for name, spec in FEATURES.items():
        values = frame[name].dropna()
        within_range = values.between(spec.minimum, spec.maximum)
        whole_numbers = values.mod(1).eq(0)
        valid = within_range & whole_numbers
        age_restricted = spec.min_age > 10 or spec.max_age < 19
        allows_structural_absence = age_restricted or spec.sexual_history or spec.female_only
        if allows_structural_absence:
            valid |= values.eq(-1)
        if not valid.all() or not np.isfinite(values).all():
            raise ValueError(f"Valores fuera del dominio de {name}")


def checked_splits(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, *, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    missing_groups = pd.Series(groups).isna().any()
    enough_groups = len(np.unique(groups)) >= folds
    if missing_groups or not enough_groups:
        raise ValueError("UPM insuficientes o ausentes para la partición solicitada")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y, groups))
    labels = np.asarray(y)
    clusters = np.asarray(groups)
    for training, validation in splits:
        shared_groups = set(clusters[training]) & set(clusters[validation])
        if shared_groups:
            raise ValueError("La partición comparte UPM")
        training_classes = set(labels[training])
        validation_classes = set(labels[validation])
        if training_classes != {0, 1} or validation_classes != {0, 1}:
            raise ValueError("Cada entrenamiento y validación debe contener ambas clases")
    return splits


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    weights: pd.Series,
    *,
    folds: int,
    trees: int,
    config: ExperimentConfig,
) -> tuple[RFECV, list[str]]:
    splits = checked_splits(X, y, groups, folds=folds, seed=config.seed)
    with config_context(enable_metadata_routing=True):
        estimator = SurveyClassifier(n_estimators=trees, random_state=config.seed, n_jobs=1)
        estimator.set_fit_request(sample_weight=True)
        scorer = make_scorer(matthews_corrcoef).set_score_request(sample_weight=False)
        selector = RFECV(
            estimator,
            step=1,
            min_features_to_select=min(config.min_features, X.shape[1]),
            cv=splits,
            scoring=scorer,
            n_jobs=config.jobs,
        )
        selector.fit(X, y, sample_weight=np.asarray(weights))
    selected_columns = X.columns[selector.support_].tolist()
    return selector, selected_columns


def create_models(config: ExperimentConfig) -> dict[str, SurveyClassifier]:
    models = {}
    for name in MODEL_NAMES:
        models[name] = SurveyClassifier(
            kind=name,
            n_estimators=config.forest_trees,
            random_state=config.seed,
            n_jobs=config.jobs,
        )
    return models


def compare_models(
    development: ModelData,
    models: dict[str, SurveyClassifier],
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: ExperimentConfig,
) -> CrossValidationResult:
    predictions = pd.DataFrame(
        {
            "row_id": development.row_ids.to_numpy(),
            "y": development.target.to_numpy(),
        }
    )
    fold_scores = []
    selections = []
    fold_assignments = np.full(len(development.target), -1, dtype=int)
    for fold, (training_positions, validation_positions) in enumerate(splits):
        LOGGER.info("Pliegue %s/%s", fold + 1, len(splits))
        training = development.take(training_positions)
        validation = development.take(validation_positions)
        selector, selected_columns = select_features(
            training.features,
            training.target,
            training.groups,
            training.weights,
            folds=config.inner_folds,
            trees=config.selection_trees,
            config=config,
        )
        selections.append({"fold": fold, "features": selected_columns, "ranking": selector.ranking_.tolist()})
        fold_assignments[validation_positions] = fold
        for name, estimator in models.items():
            model = clone(estimator)
            model.fit(training.features[selected_columns], training.target, sample_weight=training.weights)
            predicted = model.predict(validation.features[selected_columns])
            probability = model.predict_proba(validation.features[selected_columns])[:, 1]
            predictions.loc[validation_positions, f"{name}_pred"] = predicted
            predictions.loc[validation_positions, f"{name}_proba"] = probability
            fold_result = {"modelo": name, "fold": fold}
            fold_result.update(metrics(validation.target, predicted, probability))
            fold_scores.append(fold_result)
    return CrossValidationResult(
        scores=pd.DataFrame(fold_scores),
        predictions=predictions,
        selections=selections,
        fold_assignments=fold_assignments,
    )


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for name in MODEL_NAMES:
        model_scores = scores.loc[scores["modelo"].eq(name)]
        metric_values = model_scores.drop(columns=["modelo", "fold"])
        summary = {"modelo": name}
        for metric in metric_values:
            summary[f"{metric}_media"] = float(metric_values[metric].mean())
            summary[f"{metric}_sd"] = float(metric_values[metric].std(ddof=0))
        summaries.append(summary)
    comparison = pd.DataFrame(summaries)
    return comparison.sort_values("MCC_media", ascending=False, kind="stable")


def evaluate_without_selection(
    development: ModelData,
    estimator: SurveyClassifier,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    results = []
    for fold, (training_positions, validation_positions) in enumerate(splits):
        training = development.take(training_positions)
        validation = development.take(validation_positions)
        model = clone(estimator)
        model.fit(training.features, training.target, sample_weight=training.weights)
        predicted = model.predict(validation.features)
        probability = model.predict_proba(validation.features)[:, 1]
        fold_result = {"fold": fold}
        fold_result.update(metrics(validation.target, predicted, probability))
        results.append(fold_result)
    return pd.DataFrame(results)


def selection_frequency(selections: list[dict]) -> pd.DataFrame:
    frequencies = {}
    for feature in FEATURES:
        selected_folds = 0
        for selection in selections:
            if feature in selection["features"]:
                selected_folds += 1
        frequencies[feature] = selected_folds / len(selections)
    frequency_series = pd.Series(frequencies)
    return frequency_series.rename_axis("feature").reset_index(name="selection_frequency")


def fit_and_evaluate(
    development: ModelData,
    testing: ModelData,
    model_name: str,
    selected_columns: list[str],
    config: ExperimentConfig,
) -> TestEvaluation:
    model = prediction_pipeline(
        selected_columns,
        model_name,
        trees=config.forest_trees,
        seed=config.seed,
        jobs=config.jobs,
    )
    model.fit(development.features[selected_columns], development.target, model__sample_weight=development.weights)
    test_features = testing.features[selected_columns]
    return TestEvaluation(
        model=model,
        predictions=model.predict(test_features),
        probabilities=model.predict_proba(test_features)[:, 1],
    )


def subgroup_evaluations(testing: ModelData, evaluation: TestEvaluation, config: ExperimentConfig) -> dict:
    target = testing.target.to_numpy()
    groups = testing.groups.to_numpy()
    subgroups = {}
    for asi_value in (0, 1):
        belongs_to_group = testing.features["asi"].to_numpy() == asi_value
        if not belongs_to_group.any():
            continue
        group_target = target[belongs_to_group]
        group_predictions = evaluation.predictions[belongs_to_group]
        group_probabilities = evaluation.probabilities[belongs_to_group]
        group_info = {
            "n": int(belongs_to_group.sum()),
            "n_positive": int(group_target.sum()),
        }
        group_info.update(metrics(group_target, group_predictions, group_probabilities))
        group_info["incertidumbre"] = group_bootstrap(
            group_target,
            group_predictions,
            group_probabilities,
            groups[belongs_to_group],
            repetitions=config.bootstrap_repetitions,
            seed=config.seed,
        )
        subgroups[str(asi_value)] = group_info
    return subgroups


def build_model_info(
    *,
    model_name: str,
    selected_columns: list[str],
    comparison: pd.DataFrame,
    development: ModelData,
    testing: ModelData,
    evaluation: TestEvaluation,
    config: ExperimentConfig,
) -> dict:
    test_metrics = metrics(testing.target, evaluation.predictions, evaluation.probabilities)
    weighted_metrics = metrics(testing.target, evaluation.predictions, evaluation.probabilities, testing.weights)
    uncertainty = group_bootstrap(
        testing.target,
        evaluation.predictions,
        evaluation.probabilities,
        testing.groups,
        repetitions=config.bootstrap_repetitions,
        seed=config.seed,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mejor_modelo": model_name,
        "target": TARGET,
        "features": selected_columns,
        "clases": [0, 1],
        "n_dev": len(development.target),
        "n_test": len(testing.target),
        "config": asdict(config),
        "metricas_cv": comparison.iloc[0].to_dict(),
        "metricas_holdout": test_metrics,
        "metricas_holdout_ponderadas": weighted_metrics,
        "subgrupos_asi": subgroup_evaluations(testing, evaluation, config),
        "incertidumbre": uncertainty,
        "evaluation_scope": "internal_group_holdout",
        "test_history": "previously_inspected_internal_partition",
        "selection_metric": "unweighted_mcc",
        "training_weights": "survey_weights_balanced_within_training",
        "external_probability_calibration": False,
        "feature_representation": "value + not_applicable_indicator + unknown_indicator",
        "n_model_channels": len(FEATURE_CHANNELS) * len(selected_columns),
        "not_applicable_value": -1,
        "model_role": "full_data_demonstration",
        "evaluated_model": "modelo_evaluado.joblib",
    }


def selection_curve(selector: RFECV) -> pd.DataFrame:
    curve_columns = {}
    for name, values in selector.cv_results_.items():
        if np.asarray(values).ndim == 1:
            curve_columns[name] = values
    return pd.DataFrame(curve_columns)


def save_validation_results(
    workspace: Path,
    frame: pd.DataFrame,
    development_positions: np.ndarray,
    validation: CrossValidationResult,
    comparison: pd.DataFrame,
    baseline: pd.DataFrame,
) -> None:
    output = workspace / "clasificacion"
    output.mkdir(parents=True, exist_ok=True)
    partitions = frame[["row_id", "upm", "estrato"]].copy()
    partitions["partition"] = "test"
    partitions.loc[development_positions, "partition"] = "development"
    partitions["fold"] = -1
    partitions.loc[development_positions, "fold"] = validation.fold_assignments
    partitions.to_csv(output / "particiones.csv", index=False)
    validation.predictions.to_csv(output / "predicciones_oof.csv", index=False)
    validation.scores.to_csv(output / "metricas_por_fold.csv", index=False)
    comparison.to_csv(output / "comparacion_modelos.csv", index=False)
    baseline.to_csv(output / "baseline_sin_seleccion.csv", index=False)


def save_selection_results(
    workspace: Path,
    selector: RFECV,
    feature_names: pd.Index,
    selected_columns: list[str],
    selections: list[dict],
) -> pd.DataFrame:
    output = workspace / "seleccion"
    output.mkdir(parents=True, exist_ok=True)
    ranking = pd.DataFrame({"feature": feature_names, "ranking": selector.ranking_, "seleccionada": selector.support_})
    ranking.to_csv(output / "ranking_features.csv", index=False)
    curve = selection_curve(selector)
    curve.to_csv(output / "curva_rfecv.csv", index=False)
    selection_frequency(selections).to_csv(output / "estabilidad_seleccion.csv", index=False)
    write_json(output / "seleccion_por_fold.json", selections)
    write_json(
        output / "features_seleccionadas.json",
        {
            "target": TARGET,
            "n_features": len(selected_columns),
            "features": selected_columns,
        },
    )
    return curve


def save_test_results(
    workspace: Path,
    test_frame: pd.DataFrame,
    testing: ModelData,
    evaluation: TestEvaluation,
    final_model: Pipeline,
    model_info: dict,
) -> None:
    output = workspace / "clasificacion"
    output.mkdir(parents=True, exist_ok=True)
    dump(evaluation.model, output / "modelo_evaluado.joblib")
    dump(final_model, output / "mejor_modelo.joblib")
    predictions = test_frame[["row_id", "upm", "ponde_f", "asi"]].copy()
    predictions["y"] = testing.target
    predictions["prediction"] = evaluation.predictions
    predictions["probability"] = evaluation.probabilities
    predictions.to_csv(output / "predicciones_prueba.csv", index=False)
    calibration = calibration_table(testing.target, evaluation.probabilities)
    calibration.to_csv(output / "calibracion_prueba.csv", index=False)
    write_json(output / "mejor_modelo_info.json", model_info)


def train(frame: pd.DataFrame, workspace: Path, config: ExperimentConfig) -> dict:
    validate_ml(frame)
    data = ModelData.from_frame(frame)
    holdout_splits = checked_splits(
        data.features,
        data.target,
        data.groups,
        folds=config.outer_folds,
        seed=config.seed,
    )
    development_positions, test_positions = holdout_splits[0]
    development = data.take(development_positions)
    testing = data.take(test_positions)
    validation_splits = checked_splits(
        development.features,
        development.target,
        development.groups,
        folds=config.outer_folds,
        seed=config.seed,
    )
    models = create_models(config)
    validation = compare_models(development, models, validation_splits, config)
    comparison = summarize_scores(validation.scores)
    winner = comparison.iloc[0]["modelo"]
    baseline = evaluate_without_selection(development, models[winner], validation_splits)

    LOGGER.info("Selección final de variables")
    selector, selected_columns = select_features(
        development.features,
        development.target,
        development.groups,
        development.weights,
        folds=config.final_folds,
        trees=config.final_selection_trees,
        config=config,
    )
    evaluation = fit_and_evaluate(development, testing, winner, selected_columns, config)
    final_model = clone(evaluation.model)
    final_model.fit(data.features[selected_columns], data.target, model__sample_weight=data.weights)
    model_info = build_model_info(
        model_name=winner,
        selected_columns=selected_columns,
        comparison=comparison,
        development=development,
        testing=testing,
        evaluation=evaluation,
        config=config,
    )

    save_validation_results(workspace, frame, development_positions, validation, comparison, baseline)
    curve = save_selection_results(workspace, selector, data.features.columns, selected_columns, validation.selections)
    save_test_results(workspace, frame.iloc[test_positions], testing, evaluation, final_model, model_info)
    from suicide_predictors.plots import classification_figures

    classification_figures(comparison, validation.predictions, curve, len(selected_columns), workspace)
    return model_info
