import pandas as pd
import streamlit as st

from dashboard.data import artifact_root, load_comparacion_modelos, load_mejor_modelo_info
from dashboard.presentation import render_figure, render_table

METRIC_LABELS = {
    "MCC": "MCC",
    "Balanced accuracy": "balanced_accuracy",
    "F1 macro": "f1_macro",
    "Sensibilidad": "sensibilidad",
    "Especificidad": "especificidad",
    "PPV": "ppv",
    "AUC-ROC": "auc_roc",
}


def comparison_table(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, model_scores in comparison.iterrows():
        row = {"Modelo": model_scores["modelo"]}
        for label, metric in METRIC_LABELS.items():
            mean = model_scores[f"{metric}_media"]
            standard_deviation = model_scores[f"{metric}_sd"]
            row[label] = f"{mean:.3f} ± {standard_deviation:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def holdout_table(info: dict) -> pd.DataFrame:
    intervals = info.get("incertidumbre", {}).get("intervals", {})
    rows = []
    for label, metric in METRIC_LABELS.items():
        value = info["metricas_holdout"].get(metric)
        interval = intervals.get(metric)
        if interval:
            interval_label = f"[{interval['lower']:.3f}, {interval['upper']:.3f}]"
        else:
            interval_label = "No disponible"
        rows.append({"Métrica": label, "Valor": value, "IC 95% por UPM": interval_label})
    return pd.DataFrame(rows)


def show_figures() -> None:
    directory = artifact_root() / "figuras"
    figures = (
        ("MCC", "clasificacion/17_comparacion_mcc.png", "MCC medio por algoritmo."),
        (
            "Matrices",
            "clasificacion/16_matrices_confusion.png",
            "Matrices de confusión de predicciones fuera de pliegue.",
        ),
        ("ROC", "clasificacion/18_curvas_roc.png", "Curvas ROC y AUC de predicciones agrupadas fuera de pliegue."),
        ("Selección", "seleccion/15_rfecv_seleccion.png", "Selección de predictores mediante RFECV sobre desarrollo."),
    )
    tabs = st.tabs([label for label, _, _ in figures])
    for tab, (_, filename, caption) in zip(tabs, figures, strict=True):
        with tab:
            render_figure(directory / filename, caption)


def render() -> None:
    st.subheader("Comparación de clasificadores")
    comparison = load_comparacion_modelos()
    info = load_mejor_modelo_info()
    if comparison.empty or not info:
        st.warning("Ejecuta el pipeline para generar la evaluación.")
        return
    st.metric("Algoritmo seleccionado por MCC medio", info["mejor_modelo"])
    cv_tab, test_tab = st.tabs(["Validación cruzada", "Prueba interna"])
    with cv_tab:
        render_table(comparison_table(comparison))
        st.caption("Media y desviación entre pliegues de desarrollo; métricas no ponderadas.")
    with test_tab:
        if info.get("metricas_holdout"):
            render_table(holdout_table(info))
        else:
            st.info("No hay resultados de prueba interna en esta ejecución.")
    st.subheader("Figuras de evaluación y selección")
    show_figures()
    with st.expander("Configuración del experimento"):
        development_count = info.get("n_dev", "N/D")
        test_count = info.get("n_test", "N/D")
        st.write(f"Desarrollo: {development_count}; prueba: {test_count} participantes.")
        st.write(f"Predictores del modelo final: {len(info['features'])}.")
        st.write("La selección se realiza dentro de cada entrenamiento externo y puede variar entre pliegues.")
        if "config" in info:
            st.json(info["config"])
        st.caption("La evaluación es interna. No acredita validación prospectiva ni utilidad clínica.")
