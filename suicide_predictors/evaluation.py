import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics(
    y: pd.Series | np.ndarray,
    predicted: np.ndarray,
    probability: np.ndarray,
    weights: pd.Series | np.ndarray | None = None,
) -> dict[str, float | None]:
    target = np.asarray(y)
    predicted = np.asarray(predicted)
    probability = np.asarray(probability)
    both_classes_present = len(np.unique(target)) == 2
    result = {
        "MCC": matthews_corrcoef(target, predicted, sample_weight=weights),
        "balanced_accuracy": None,
        "f1_macro": f1_score(target, predicted, average="macro", labels=[0, 1], zero_division=0, sample_weight=weights),
        "sensibilidad": recall_score(target, predicted, zero_division=0, sample_weight=weights),
        "especificidad": recall_score(target, predicted, pos_label=0, zero_division=0, sample_weight=weights),
        "ppv": precision_score(target, predicted, zero_division=0, sample_weight=weights),
        "auc_roc": None,
        "average_precision": None,
        "brier": brier_score_loss(target, probability, sample_weight=weights),
    }
    if both_classes_present:
        result["balanced_accuracy"] = balanced_accuracy_score(target, predicted, sample_weight=weights)
        result["auc_roc"] = roc_auc_score(target, probability, sample_weight=weights)
        result["average_precision"] = average_precision_score(target, probability, sample_weight=weights)
    cleaned = {}
    for name, value in result.items():
        if value is None or not np.isfinite(value):
            cleaned[name] = None
        else:
            cleaned[name] = float(value)
    return cleaned


def bootstrap_intervals(draws: list[dict]) -> dict:
    if not draws:
        return {}
    intervals = {}
    for metric in draws[0]:
        valid_values = []
        for draw in draws:
            value = draw[metric]
            if value is not None:
                valid_values.append(value)
        if valid_values:
            lower = float(np.quantile(valid_values, 0.025))
            upper = float(np.quantile(valid_values, 0.975))
            intervals[metric] = {"lower": lower, "upper": upper}
    return intervals


def group_bootstrap(
    y: pd.Series | np.ndarray,
    predicted: np.ndarray,
    probability: np.ndarray,
    groups: pd.Series | np.ndarray,
    *,
    repetitions: int = 500,
    seed: int = 42,
) -> dict:
    target = np.asarray(y)
    predicted = np.asarray(predicted)
    probability = np.asarray(probability)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    positions_by_group = {}
    for group in unique_groups:
        positions_by_group[group] = np.flatnonzero(groups == group)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sampled_groups = rng.choice(unique_groups, len(unique_groups), replace=True)
        sampled_positions = []
        for group in sampled_groups:
            sampled_positions.append(positions_by_group[group])
        indices = np.concatenate(sampled_positions)
        sampled_target = target[indices]
        if len(np.unique(sampled_target)) != 2:
            continue
        draw = metrics(sampled_target, predicted[indices], probability[indices])
        draws.append(draw)
    return {
        "method": "percentile_bootstrap_upm",
        "requested": repetitions,
        "valid": len(draws),
        "intervals": bootstrap_intervals(draws),
    }


def calibration_table(y: pd.Series | np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"observed": np.asarray(y), "probability": np.asarray(probability)})
    bin_edges = np.linspace(0, 1, bins + 1)
    frame["bin"] = pd.cut(frame["probability"], bin_edges, include_lowest=True)
    grouped = frame.groupby("bin", observed=True)
    summary = grouped.agg(
        n=("observed", "size"),
        observed_rate=("observed", "mean"),
        mean_probability=("probability", "mean"),
    )
    return summary.reset_index()
