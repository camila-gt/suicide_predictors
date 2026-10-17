import pandas as pd
import streamlit as st


def _wprev(df: pd.DataFrame, col: str) -> float:
    """Prevalencia ponderada (%) usando el factor de expansión."""
    valid = df[df[col].notna()]
    if valid.empty or valid["ponderador"].sum() == 0:
        return 0.0
    return valid.loc[valid[col] == 1, "ponderador"].sum() / valid["ponderador"].sum() * 100


def render(df: pd.DataFrame) -> None:
    st.title("Abuso sexual infantil y conductas suicidas en adolescentes")
    st.subheader("ENSANUT Continua 2024 · Estudio observacional transversal")

    st.markdown(
        """
        Este dashboard presenta los resultados de un análisis de la relación entre el **abuso
        sexual infantil (ASI)** y las **conductas autolíticas** (ideación e intento suicida)
        en adolescentes mexicanos de 10 a 19 años, utilizando datos de la
        **Encuesta Nacional de Salud y Nutrición (ENSANUT) Continua 2024**.

        El estudio evalúa la prevalencia ponderada de ASI e indicadores de conductas suicidas,
        estimada con el factor de expansión muestral de la ENSANUT para reflejar la
        población nacional de adolescentes.

        **Navegación del dashboard:**
        - **Muestra** — Prevalencias ponderadas y desgloses por sexo y grupo de edad.
        - **Métodos de intento** — Frecuencias muestrales de los métodos reportados.
        - **Modelos de ML** — Comparación de cuatro clasificadores y evaluación interna.
        - **Interpretabilidad SHAP** — Contribuciones de los predictores a la salida del modelo.
        - **Predicción interactiva** — Simulador académico del antecedente de intento con perfiles ficticios.
        - **Mapa geográfico** — Distribución territorial de casos de ASI en la República Mexicana.
        """
    )

    st.divider()

    n_total = len(df)
    prev_asi = _wprev(df, "asi")
    prev_intento = _wprev(df, "intento")
    prev_ideacion = _wprev(df, "ideacion")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Adolescentes en la muestra", f"{n_total:,}", border=True)
    with col_m2:
        st.metric("Prevalencia ponderada de ASI", f"{prev_asi:.1f}%", border=True)
    with col_m3:
        st.metric("Prevalencia ponderada de ideación suicida", f"{prev_ideacion:.1f}%", border=True)
    with col_m4:
        st.metric("Prevalencia ponderada de intento de suicidio", f"{prev_intento:.1f}%", border=True)

    st.caption(
        "Las prevalencias se calculan con el factor de expansión muestral (ponderador) "
        "de la ENSANUT 2024 para representar estimaciones poblacionales. "
        "Los tamaños de muestra (n) se reportan sin ponderar."
    )
