from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from suicide_predictors.artifacts import file_hash, write_json
from suicide_predictors.plots import shap_figures


def explain(
    workspace: Path, *, sample_size: int = 200, background_size: int = 50, seed: int = 0, permutations: int = 5
) -> dict:
    import shap

    if min(sample_size, background_size, permutations) < 1:
        raise ValueError("Las muestras y las permutaciones deben ser positivas")
    model_path = workspace / "clasificacion" / "mejor_modelo.joblib"
    model = load(model_path)
    columns = list(model.named_steps["columns"].columns)
    data = pd.read_csv(workspace / "preprocesamiento" / "adolescentes_ml.csv")
    X = data[columns].copy()
    sample = shap.utils.sample(X, min(sample_size, len(X)), random_state=seed)
    background = shap.utils.sample(X, min(background_size, len(X)), random_state=seed)

    def positive_probability(values: pd.DataFrame | np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(values, columns=columns))[:, 1]

    explainer = shap.Explainer(positive_probability, background, algorithm="permutation", seed=seed)
    evaluations_per_permutation = 2 * len(columns) + 1
    max_evaluations = evaluations_per_permutation * permutations
    explanation = explainer(sample, max_evals=max_evaluations)
    values = np.asarray(explanation.values)
    reconstructed_probability = explanation.base_values + values.sum(axis=1)
    model_probability = positive_probability(sample)
    if not np.allclose(reconstructed_probability, model_probability, atol=1e-6):
        raise ValueError("Las contribuciones SHAP no reconstruyen la probabilidad explicada")
    output = workspace / "shap"
    output.mkdir(parents=True, exist_ok=True)
    mean_absolute_contribution = np.abs(values).mean(axis=0)
    importance = pd.Series(mean_absolute_contribution, index=columns)
    importance = importance.sort_values(ascending=False)
    importance_table = importance.rename_axis("feature").reset_index(name="importancia_shap")
    importance_table.to_csv(output / "shap_importancia.csv", index=False)
    np.savez_compressed(
        output / "explicaciones.npz",
        values=values,
        base_values=explanation.base_values,
        data=sample.to_numpy(),
        row_id=data.loc[sample.index, "row_id"].to_numpy(),
        features=np.asarray(columns),
        background_row_id=data.loc[background.index, "row_id"].to_numpy(),
    )
    shap_figures(
        importance,
        explanation,
        workspace / "shap",
        model_name=model.named_steps["model"].kind,
    )
    info = {
        "explainer": type(explainer).__name__,
        "output": "positive_class_probability",
        "seed": seed,
        "sample_size": len(sample),
        "background_size": len(background),
        "permutations": permutations,
        "model_sha256": file_hash(model_path),
        "weighted_summary": False,
        "missingness": "NaN=unknown; -1=not_applicable; explanations_include_missingness_channels",
    }
    write_json(output / "explanation_info.json", info)
    return info
