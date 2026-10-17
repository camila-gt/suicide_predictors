"""Presentación acotada de resultados, sin modificar los artefactos del pipeline."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from dashboard.config import CAT_BASE, CAT_TEXT

IMAGE_MAX_WIDTH = 900
IMAGE_MAX_HEIGHT = 560
TABLE_MAX_HEIGHT = 360
TABLE_ROW_HEIGHT = 34


def fitted_image_width(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        raise ValueError("Las dimensiones de la figura deben ser positivas")
    scale = min(1.0, IMAGE_MAX_WIDTH / width, IMAGE_MAX_HEIGHT / height)
    return max(1, int(width * scale))


def table_height(rows: int) -> int:
    return min(TABLE_MAX_HEIGHT, 40 + max(rows, 1) * TABLE_ROW_HEIGHT)


def render_figure(path: Path, caption: str) -> None:
    if not path.is_file():
        st.info("Esta figura no está disponible en la ejecución activa.")
        return
    with Image.open(path) as image:
        width, height = image.size
    preview_width = fitted_image_width(width, height)
    with st.container(horizontal_alignment="center"):
        st.image(str(path), caption=caption, width=preview_width)
    with st.container(horizontal=True, gap="small"):
        st.download_button(
            "Descargar PNG original",
            data=path.read_bytes(),
            file_name=path.name,
            mime="image/png",
            key=f"figure_png_{path.stem}",
            on_click="ignore",
        )
        pdf = path.with_suffix(".pdf")
        if pdf.is_file():
            st.download_button(
                "Descargar PDF",
                data=pdf.read_bytes(),
                file_name=pdf.name,
                mime="application/pdf",
                key=f"figure_pdf_{path.stem}",
                on_click="ignore",
            )


def render_table(data: pd.DataFrame, *, hide_index: bool = True) -> None:
    st.dataframe(
        data,
        hide_index=hide_index,
        width="stretch",
        height=table_height(len(data)),
        row_height=TABLE_ROW_HEIGHT,
    )


def render_plotly(figure: go.Figure, *, height: int = 360) -> None:
    chart = go.Figure(figure)
    title = chart.layout.title.text
    if title:
        st.markdown(f"**{title}**")
    chart.update_layout(
        title=None,
        autosize=True,
        width=None,
        height=height,
        paper_bgcolor=CAT_BASE,
        plot_bgcolor=CAT_BASE,
        font={"color": CAT_TEXT, "size": 12},
        margin={"l": 12, "r": 20, "t": 36, "b": 36},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    chart.update_xaxes(automargin=True)
    chart.update_yaxes(automargin=True)
    st.plotly_chart(
        chart,
        width="stretch",
        height=height,
        theme=None,
        config={"responsive": True, "displayModeBar": False, "scrollZoom": False},
    )
