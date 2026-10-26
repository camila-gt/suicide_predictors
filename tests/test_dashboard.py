import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from streamlit.testing.v1 import AppTest

from dashboard.presentation import (
    IMAGE_MAX_HEIGHT,
    IMAGE_MAX_WIDTH,
    TABLE_MAX_HEIGHT,
    fitted_image_width,
    render_figure,
    render_plotly,
    render_table,
    table_height,
)
from dashboard.tabs.modelos_ml import METRIC_LABELS


def write_images(workspace: Path, relative_paths: list[str]) -> None:
    for relative in relative_paths:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new("RGB", (1200, 900), "white") as image:
            image.save(path)


class DashboardPresentationTests(unittest.TestCase):
    def test_preview_fits_both_limits_without_changing_aspect_ratio(self):
        for width, height in ((2400, 1500), (3600, 3000), (1800, 2400), (4000, 1000)):
            with self.subTest(size=(width, height)):
                displayed_width = fitted_image_width(width, height)
                displayed_height = displayed_width * height / width
                self.assertLessEqual(displayed_width, IMAGE_MAX_WIDTH)
                self.assertLessEqual(displayed_height, IMAGE_MAX_HEIGHT)
                self.assertLessEqual(displayed_width, width)

    def test_small_image_is_not_upscaled(self):
        self.assertEqual(fitted_image_width(320, 200), 320)

    def test_invalid_image_dimensions_are_rejected(self):
        for width, height in ((0, 10), (10, 0), (-1, 10)):
            with self.subTest(size=(width, height)), self.assertRaises(ValueError):
                fitted_image_width(width, height)

    def test_tables_fit_their_rows_but_have_a_height_limit(self):
        self.assertEqual(table_height(0), table_height(1))
        self.assertLess(table_height(4), table_height(7))
        self.assertEqual(table_height(100), TABLE_MAX_HEIGHT)

    def test_table_renderer_preserves_data_and_caps_height(self):
        data = pd.DataFrame({"value": range(100)})
        with patch("dashboard.presentation.st.dataframe") as display:
            render_table(data)
        self.assertIs(display.call_args.args[0], data)
        self.assertEqual(display.call_args.kwargs["height"], TABLE_MAX_HEIGHT)
        self.assertEqual(display.call_args.kwargs["width"], "stretch")

    def test_image_preview_and_downloads_do_not_modify_original_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "figure.png"
            with Image.new("RGB", (2400, 1800), "white") as image:
                image.save(path)
            original_png = path.read_bytes()
            with (
                patch("dashboard.presentation.st.container"),
                patch("dashboard.presentation.st.image") as display,
                patch("dashboard.presentation.st.download_button") as download,
            ):
                render_figure(path, "Figura de prueba")
            self.assertEqual(display.call_args.kwargs["width"], fitted_image_width(2400, 1800))
            self.assertEqual(download.call_args_list[0].kwargs["data"], original_png)
            download.assert_called_once()
            self.assertEqual(path.read_bytes(), original_png)

    def test_missing_figure_is_reported_without_reading_an_image(self):
        with (
            patch("dashboard.presentation.st.info") as message,
            patch("dashboard.presentation.st.image") as image,
        ):
            render_figure(Path("does-not-exist.png"), "Figura")
        message.assert_called_once()
        image.assert_not_called()

    def test_plotly_renderer_preserves_data_and_moves_title_outside_the_chart(self):
        figure = go.Figure(go.Bar(x=["A", "B"], y=[1, 2], marker_color="#89b4fa"))
        figure.update_layout(title="Título largo que debe poder ajustarse", width=2000, height=900)
        original = figure.to_json()
        with (
            patch("dashboard.presentation.st.markdown") as heading,
            patch("dashboard.presentation.st.plotly_chart") as display,
        ):
            render_plotly(figure, height=320)
        rendered = display.call_args.args[0]
        self.assertIsNone(rendered.layout.width)
        self.assertIsNone(rendered.layout.title.text)
        self.assertEqual(rendered.layout.height, 320)
        self.assertEqual(list(rendered.data[0].y), [1, 2])
        self.assertEqual(rendered.data[0].marker.color, "#89b4fa")
        self.assertTrue(display.call_args.kwargs["config"]["responsive"])
        self.assertEqual(figure.to_json(), original)
        heading.assert_called_once()


class DashboardPageTests(unittest.TestCase):
    def test_model_page_separates_figures_into_tabs(self):
        script = """
from pathlib import Path
from unittest.mock import patch
import streamlit as st
from dashboard.tabs import modelos_ml
with (
    patch.object(modelos_ml, 'artifact_root', return_value=Path(st.session_state['workspace'])),
    patch.object(modelos_ml, 'load_comparacion_modelos', return_value=st.session_state['comparison']),
    patch.object(modelos_ml, 'load_mejor_modelo_info', return_value=st.session_state['info']),
):
    modelos_ml.render()
"""
        comparison = {"modelo": ["Regresión Logística", "Random Forest"]}
        for metric in METRIC_LABELS.values():
            comparison[f"{metric}_media"] = [0.5, 0.4]
            comparison[f"{metric}_sd"] = [0.01, 0.02]
        info = {
            "mejor_modelo": "Regresión Logística",
            "features": ["edad", "asi"],
            "metricas_holdout": dict.fromkeys(METRIC_LABELS.values(), 0.5),
        }
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            write_images(
                workspace,
                [
                    "clasificacion/17_comparacion_mcc.png",
                    "clasificacion/16_matrices_confusion.png",
                    "clasificacion/18_curvas_roc.png",
                    "seleccion/15_rfecv_seleccion.png",
                ],
            )
            app = AppTest.from_string(script, default_timeout=30)
            app.session_state["workspace"] = str(workspace)
            app.session_state["comparison"] = pd.DataFrame(comparison)
            app.session_state["info"] = info
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(
                [tab.label for tab in app.tabs],
                ["Validación cruzada", "Prueba interna", "MCC", "Matrices", "ROC", "Selección"],
            )
            self.assertEqual(len(app.dataframe), 2)
            self.assertEqual(len(app.get("download_button")), 4)

    def test_shap_page_has_separate_figures_and_optional_table(self):
        script = """
from pathlib import Path
from unittest.mock import patch
import streamlit as st
from dashboard.tabs import shap_interpretabilidad as page
with (
    patch.object(page, 'artifact_root', return_value=Path(st.session_state['workspace'])),
    patch.object(page, 'load_shap_importancia', return_value=st.session_state['importance']),
    patch.object(page, 'load_mejor_modelo_info', return_value={'mejor_modelo': 'Regresión Logística'}),
):
    page.render()
"""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            write_images(workspace, ["shap/18_shap_importancia_global.png", "shap/19_shap_beeswarm.png"])
            app = AppTest.from_string(script, default_timeout=30)
            app.session_state["workspace"] = str(workspace)
            app.session_state["importance"] = pd.DataFrame(
                {"feature": ["asi", "edad"], "importancia_shap": [0.1, 0.03]}
            )
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual([tab.label for tab in app.tabs], ["Importancia global", "Beeswarm"])
            self.assertFalse(app.expander[0].proto.expanded)
            self.assertEqual(len(app.get("download_button")), 2)


if __name__ == "__main__":
    unittest.main()
