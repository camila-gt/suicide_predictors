import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.config import AGRESOR_LBL, ATENCION_LBL, COLOR_CON_ASI, COLOR_SIN_ASI, RAZON_LBL
from dashboard.presentation import render_plotly, render_table

W = "ponderador"


def _weighted_prev(sub: pd.DataFrame, col: str) -> float:
    """Prevalencia ponderada (%) de una variable binaria en un subconjunto."""
    valid = sub[sub[col].notna()]
    if valid.empty or valid[W].sum() == 0:
        return 0.0
    return valid.loc[valid[col] == 1, W].sum() / valid[W].sum() * 100


def render(df: pd.DataFrame) -> None:
    template = "plotly_dark"

    def _tabla_etiquetas(serie: pd.Series, labels: dict) -> pd.DataFrame:
        serie_lbl = serie.map(labels).dropna()
        if serie_lbl.empty:
            return pd.DataFrame(columns=["Categoria", "n", "% muestral"])
        counts = serie_lbl.value_counts().rename_axis("Categoria").reset_index(name="n")
        total = counts["n"].sum()
        counts["% muestral"] = (counts["n"] / total * 100).round(1)
        return counts

    st.info(
        "Prevalencias ponderadas con el factor de expansión muestral de la ENSANUT 2024. "
        "Los tamaños de muestra (n) y los porcentajes de las tablas descriptivas son muestrales, sin ponderar."
    )

    desenlace = st.radio(
        "Desenlace a visualizar",
        ["Ideación suicida", "Intento de suicidio"],
        horizontal=True,
    )

    col_var = "ideacion" if desenlace == "Ideación suicida" else "intento"
    df_f = df
    etiq = f"Muestra total (n = {len(df_f):,})"

    if df_f.empty:
        st.info("No hay datos para la selección actual.")
        return

    # Prevalencias ponderadas por grupo ASI
    stats_rows = []
    for label in df_f["asi_label"].dropna().unique():
        sub = df_f[df_f["asi_label"] == label]
        pct = _weighted_prev(sub, col_var)
        n = len(sub)
        stats_rows.append({"asi_label": label, "prev_pct": pct, "n": n})
    stats = pd.DataFrame(stats_rows)
    stats["bar_text"] = stats.apply(lambda r: f"{r['prev_pct']:.1f}%  (n={int(r['n'])})", axis=1)

    fig = px.bar(
        stats,
        x="asi_label",
        y="prev_pct",
        color="asi_label",
        color_discrete_map={"Con ASI": COLOR_CON_ASI, "Sin ASI": COLOR_SIN_ASI},
        text="bar_text",
        labels={"asi_label": "Grupo", "prev_pct": "Prevalencia ponderada (%)"},
        title=f"{desenlace} según ASI · {etiq}",
        height=450,
        template=template,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        yaxis_range=[0, stats["prev_pct"].max() * 1.4 + 3],
    )

    render_plotly(fig)

    st.subheader("Desglose por sexo y grupo de edad")
    col_a, col_b = st.columns(2)

    with col_a, st.container(border=True):
        rows_sexo = []
        for sexo in df_f["sexo"].unique():
            for label in df_f["asi_label"].dropna().unique():
                sub = df_f[(df_f["sexo"] == sexo) & (df_f["asi_label"] == label)]
                pct = _weighted_prev(sub, col_var)
                rows_sexo.append({"sexo": sexo, "asi_label": label, "prev_pct": pct})
        stats_sexo = pd.DataFrame(rows_sexo)
        fig2 = px.bar(
            stats_sexo,
            x="sexo",
            y="prev_pct",
            color="asi_label",
            barmode="group",
            color_discrete_map={"Con ASI": COLOR_CON_ASI, "Sin ASI": COLOR_SIN_ASI},
            labels={"sexo": "Sexo", "prev_pct": "Prevalencia ponderada (%)", "asi_label": "Grupo"},
            title=f"{desenlace} por sexo (ponderada)",
            height=320,
            template=template,
        )
        fig2.update_layout(margin={"l": 40, "r": 20, "t": 50, "b": 40})
        render_plotly(fig2, height=320)

    with col_b, st.container(border=True):
        rows_edad = []
        for ge in ["10-12", "13-15", "16-19"]:
            for label in df_f["asi_label"].dropna().unique():
                sub = df_f[(df_f["grupo_edad"] == ge) & (df_f["asi_label"] == label)]
                pct = _weighted_prev(sub, col_var)
                rows_edad.append({"grupo_edad": ge, "asi_label": label, "prev_pct": pct})
        stats_edad = pd.DataFrame(rows_edad)
        fig3 = px.bar(
            stats_edad,
            x="grupo_edad",
            y="prev_pct",
            color="asi_label",
            barmode="group",
            color_discrete_map={"Con ASI": COLOR_CON_ASI, "Sin ASI": COLOR_SIN_ASI},
            category_orders={"grupo_edad": ["10-12", "13-15", "16-19"]},
            labels={
                "grupo_edad": "Grupo de edad",
                "prev_pct": "Prevalencia ponderada (%)",
                "asi_label": "Grupo",
            },
            title=f"{desenlace} por grupo de edad (ponderada)",
            height=320,
            template=template,
        )
        fig3.update_xaxes(type="category")
        fig3.update_layout(margin={"l": 40, "r": 20, "t": 50, "b": 40})
        render_plotly(fig3, height=320)

    st.divider()

    asi_df = df[df["asi"] == 1].copy()
    with st.container(border=True):
        st.subheader("ASI: agresor, atencion y razones de no denuncia")
        fila_sup_a, fila_sup_b = st.columns(2, gap="large")

        with fila_sup_a:
            st.markdown("**Agresor**")
            tabla_agresor = _tabla_etiquetas(asi_df["parentesco_agresor"], AGRESOR_LBL)
            render_table(tabla_agresor)

        with fila_sup_b:
            st.markdown("**Atencion tras ASI**")
            tabla_atencion = _tabla_etiquetas(asi_df["atencion_tras_asi"], ATENCION_LBL)
            render_table(tabla_atencion)

        st.markdown("**Razon de no denuncia**")
        base_no_denuncia = asi_df[asi_df["denuncia_asi"] == 2]
        tabla_razon = _tabla_etiquetas(base_no_denuncia["razon_no_denuncia"], RAZON_LBL)
        render_table(tabla_razon)
