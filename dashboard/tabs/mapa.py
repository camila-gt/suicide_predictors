import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.config import CODIGO_A_NOMBRE
from dashboard.presentation import render_plotly, render_table


def render(df: pd.DataFrame, geojson: dict) -> None:
    template = "plotly_dark"

    st.warning("Datos muestrales sin expandir a la población. Los conteos no son representativos a nivel estatal.")
    st.subheader("Distribución de casos de ASI por entidad federativa")

    mapa_df_agrupado = df.groupby("entidad").agg(casos_asi=("asi", "sum"), total=("asi", "count")).reset_index()

    # Incluir las 32 entidades aunque no haya respondentes en la muestra
    todas_entidades = pd.DataFrame({"entidad": list(CODIGO_A_NOMBRE.keys())})
    mapa_df = todas_entidades.merge(mapa_df_agrupado, on="entidad", how="left")
    mapa_df["casos_asi"] = mapa_df["casos_asi"].fillna(0).astype(int)
    mapa_df["total"] = mapa_df["total"].fillna(0).astype(int)
    mapa_df["prevalencia_pct"] = mapa_df.apply(
        lambda r: round(r["casos_asi"] / r["total"] * 100, 1) if r["total"] > 0 else float("nan"),
        axis=1,
    )
    mapa_df["nombre"] = mapa_df["entidad"].map(CODIGO_A_NOMBRE)
    mapa_df["cvegeo"] = mapa_df["entidad"].apply(lambda x: f"{int(x):02d}")

    fig_mapa = px.choropleth(
        mapa_df,
        geojson=geojson,
        locations="cvegeo",
        featureidkey="properties.CVEGEO",
        color="prevalencia_pct",
        color_continuous_scale="Oranges",
        hover_name="nombre",
        hover_data={
            "casos_asi": True,
            "total": True,
            "prevalencia_pct": True,
            "cvegeo": False,
        },
        labels={
            "prevalencia_pct": "Prevalencia (%)",
            "casos_asi": "Casos de ASI",
            "total": "Respondentes totales",
        },
        fitbounds="locations",
        basemap_visible=False,
        title="Prevalencia muestral de ASI por entidad federativa (%)",
        height=600,
        template=template,
    )
    fig_mapa.update_layout(margin={"l": 0, "r": 0, "t": 50, "b": 0})
    fig_mapa.update_coloraxes(colorbar={"orientation": "h", "len": 0.7, "x": 0.5, "xanchor": "center", "y": -0.1})
    render_plotly(fig_mapa, height=440)

    with st.expander("Ver tabla de datos por entidad"):
        tabla_mapa = (
            mapa_df[["nombre", "casos_asi", "total", "prevalencia_pct"]]
            .rename(
                columns={
                    "nombre": "Entidad",
                    "casos_asi": "Casos de ASI",
                    "total": "Respondentes",
                    "prevalencia_pct": "Prevalencia (%)",
                }
            )
            .sort_values("Prevalencia (%)", ascending=False)
        )
        render_table(tabla_mapa)
