import streamlit as st

from dashboard.data import artifact_root, load_mejor_modelo_info, load_shap_importancia
from dashboard.presentation import render_figure, render_table


def render() -> None:
    st.subheader("Explicabilidad de la clase positiva")
    importance = load_shap_importancia()
    info = load_mejor_modelo_info()
    if importance.empty or not info:
        st.info("No hay explicaciones SHAP disponibles para esta ejecución.")
        return
    st.write(f"Modelo explicado: {info['mejor_modelo']}.")
    st.caption(
        "SHAP describe contribuciones a la salida del modelo, no efectos causales ni cambios en el riesgo futuro. "
        "El resumen corresponde a una muestra de explicación, sin ponderación poblacional."
    )
    with st.expander("Tabla de importancia de predictores"):
        render_table(importance.rename(columns={"feature": "Predictor", "importancia_shap": "Media |SHAP|"}))
    root = artifact_root() / "shap"
    figures = (
        (
            "Importancia global",
            "18_shap_importancia_global.png",
            "Media de las contribuciones absolutas por predictor.",
        ),
        ("Beeswarm", "19_shap_beeswarm.png", "Distribución de las contribuciones a la salida para la clase «Intentó»."),
    )
    tabs = st.tabs([label for label, _, _ in figures])
    for tab, (_, filename, caption) in zip(tabs, figures, strict=True):
        with tab:
            render_figure(root / filename, caption)
    with st.expander("Cómo interpretar las contribuciones"):
        st.write(
            "Un valor SHAP positivo aumenta la salida para la clase «Intentó» respecto de la referencia; "
            "uno negativo la reduce. El color representa el valor de entrada. La depresión se utiliza "
            "como puntaje CES-D continuo, no como diagnóstico binario. El valor técnico -1 indica que "
            "un reactivo no aplica al perfil según el cuestionario. Las contribuciones de un predictor "
            "incluyen también sus indicadores de desconocimiento y no aplicabilidad; no son efectos clínicos aislados."
        )
