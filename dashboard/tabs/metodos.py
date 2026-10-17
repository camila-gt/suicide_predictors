from textwrap import fill

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.config import COLOR_CON_ASI, METODO_LBL
from dashboard.presentation import render_plotly


def render(df: pd.DataFrame) -> None:
    template = "plotly_dark"

    st.subheader("Métodos de intento de suicidio")

    df_intento = df[df["intento"] == 1].copy()

    if df_intento.empty:
        st.warning("No hay casos de intento de suicidio registrados.")
        return

    etiq = "Muestra Total"
    st.caption(f"Casos de intento de suicidio · {etiq} · n = {len(df_intento)}")

    metodos_cols = [c for c in df_intento.columns if c.startswith("metodo_")]
    freq = {METODO_LBL[m]: int(df_intento[m].sum()) for m in metodos_cols if m in METODO_LBL}
    freq_df = pd.DataFrame(freq.items(), columns=["Método", "Casos"]).sort_values("Casos", ascending=True)

    fig_met = px.bar(
        freq_df,
        x="Casos",
        y="Método",
        orientation="h",
        text="Casos",
        color="Casos",
        color_continuous_scale=[[0, "#f7e5dc"], [1, COLOR_CON_ASI]],
        title=f"Frecuencia de métodos de intento de suicidio · {etiq}",
        height=480,
        template=template,
    )
    fig_met.update_traces(textposition="outside")
    fig_met.update_layout(coloraxis_showscale=False, yaxis_title="")
    wrapped_labels = [fill(label, width=22).replace("\n", "<br>") for label in freq_df["Método"]]
    fig_met.update_yaxes(tickmode="array", tickvals=freq_df["Método"], ticktext=wrapped_labels)
    render_plotly(fig_met, height=420)

    st.caption(
        "Un mismo caso puede reportar más de un método. "
        "Los valores reflejan las menciones totales, no el número de individuos."
    )
