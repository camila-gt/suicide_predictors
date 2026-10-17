import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

from suicide_predictors.plots import (
    MODEL_COLORS,
    PALETTE,
    SELECTION_COLORS,
    SEX_COLORS,
    classification_figures,
    exploration_bars,
    explore,
    percentage_label,
    plot_age_distribution,
    plot_attempt_methods,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_roc_curves,
    plot_selection_curve,
    plot_sex_distribution,
    plot_target_distribution,
    prevalence,
    prevalence_groups,
    prevalence_note,
    shap_figures,
    weighted_categories,
)


def descriptive_frame():
    return pd.DataFrame(
        {
            "edad": [10, 10, 11, 12],
            "sexo": ["Hombre", "Mujer", "Hombre", "Mujer"],
            "ponderador": [1.0, 2.0, 3.0, 4.0],
            "asi": [0, 1, 1, 0],
            "ideacion": [0, 1, np.nan, 0],
            "intento": [0, 1, 0, 1],
            "intento_suicidio": [3, 1, 3, 2],
            "violencia_reciente": [0, 1, 0, np.nan],
            "abuso_sexual": [3, 1, 2, 3],
            "parentesco_agresor": [np.nan, 2, 5, np.nan],
            "sexo_agresor": [np.nan, 1, 2, np.nan],
            "denuncia_asi": [np.nan, 1, 2, np.nan],
            "razon_no_denuncia": [np.nan, np.nan, 1, np.nan],
            "atencion_tras_asi": [np.nan, 6, 1, np.nan],
            "metodo_medicamentos": [np.nan, 1, np.nan, 0],
            "metodo_cortantes": [np.nan, 1, np.nan, 1],
        }
    )


class ExplorationPlotTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")

    def test_weighted_percentages_exclude_missing_responses(self):
        data = pd.DataFrame({"answer": [0, 1, np.nan], "ponderador": [1, 3, 100]})
        result = weighted_categories(data, "answer")
        self.assertEqual(result.to_dict(), {0: 25, 1: 75})
        self.assertEqual(prevalence(data, "answer"), 75)

    def test_percentage_annotations_keep_values_and_zero_baseline(self):
        values = pd.Series([1.26, 2.0], index=["A", "B"])
        with patch("suicide_predictors.plots.save_exploration") as save:
            exploration_bars(values, "Título", Path("unused.png"), note="n = 2.")
        figure = save.call_args.args[0]
        axes = figure.axes[0]
        np.testing.assert_equal([bar.get_width() for bar in axes.patches], [2.0, 1.26])
        self.assertEqual([label.get_text() for label in axes.get_yticklabels()], ["B", "A"])
        self.assertEqual([label.get_text() for label in axes.texts], ["2,0 %", "1,3 %"])
        self.assertEqual(axes.get_xlim()[0], 0)

    def test_small_positive_percentage_is_not_displayed_as_zero(self):
        self.assertEqual(percentage_label(0.03), "<0,1 %")
        self.assertEqual(percentage_label(0), "0,0 %")
        self.assertEqual(percentage_label(12.34), "12,3 %")

    def test_age_bars_preserve_unweighted_counts_and_do_not_overlap(self):
        with patch("suicide_predictors.plots.save_exploration") as save:
            plot_age_distribution(descriptive_frame(), Path("unused.png"))
        figure = save.call_args.args[0]
        axes = figure.axes[0]
        male, female = axes.containers
        self.assertEqual(sum(bar.get_height() for bar in male), 2)
        self.assertEqual(sum(bar.get_height() for bar in female), 2)
        self.assertLess(male[0].get_x() + male[0].get_width(), female[0].get_x())
        np.testing.assert_equal(axes.get_xticks(), np.arange(10, 20))
        self.assertIn("sin ponderar", axes.get_ylabel())

    def test_sex_prevalences_keep_order_and_group_denominators(self):
        summary = prevalence_groups(descriptive_frame(), "asi")
        self.assertEqual(summary.index.tolist(), ["Hombre", "Mujer", "Total"])
        self.assertEqual(summary["percentage"].to_dict(), {"Hombre": 75, "Mujer": 100 / 3, "Total": 50})
        note = prevalence_note(summary)
        self.assertIn("Hombre: n = 2", note)
        self.assertIn("Mujer: n = 2", note)
        self.assertIn("Total: n = 4", note)

    def test_original_sex_palette_and_pie_are_restored(self):
        self.assertEqual(SEX_COLORS, {"Hombre": "#89b4fa", "Mujer": "#f5c2e7"})
        with patch("suicide_predictors.plots.save_exploration") as save:
            plot_sex_distribution(descriptive_frame(), Path("unused.png"))
        axes = save.call_args.args[0].axes[0]
        self.assertEqual(len(axes.patches), 2)
        self.assertEqual([label.get_text() for label in axes.texts], ["Hombres", "40,0 %", "Mujeres", "60,0 %"])

    def test_method_labels_are_readable_and_counts_stay_unweighted(self):
        with patch("suicide_predictors.plots.count_comparison_panels") as panels:
            plot_attempt_methods(descriptive_frame(), Path("unused.png"))
        call = panels.call_args
        values = call.args[0]
        self.assertEqual(values.to_dict(), {"Envenenamiento con medicamentos": 1, "Objetos cortantes": 2})
        self.assertEqual(call.kwargs["color"], PALETTE["peach"])
        by_sex = call.args[1]
        self.assertEqual(by_sex.loc["Objetos cortantes"].to_dict(), {"Hombre": 0, "Mujer": 2})
        self.assertIn("n = 2", call.kwargs["note"])
        self.assertIn("no son excluyentes", call.kwargs["note"])

    def test_unknown_values_are_not_drawn_as_zero(self):
        values = pd.Series([np.nan], index=["Sin respuesta"])
        with patch("suicide_predictors.plots.save_exploration") as save:
            exploration_bars(values, "Título", Path("unused.png"), note="Sin respuestas.")
        axes = save.call_args.args[0].axes[0]
        self.assertFalse(axes.patches)
        self.assertIn("Sin respuestas disponibles", axes.texts[0].get_text())

    def test_exploration_exports_png_and_pdf_without_changing_statistics(self):
        data = descriptive_frame()
        original = data.copy(deep=True)
        original_font_size = plt.rcParams["font.size"]
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            explore(data, workspace)
            directory = workspace / "figuras/exploracion"
            png_files = sorted(directory.glob("*.png"))
            pdf_files = sorted(directory.glob("*.pdf"))
            self.assertEqual(len(png_files), 13)
            self.assertEqual(len(pdf_files), 13)
            self.assertEqual(plt.imread(png_files[0]).shape[1], 1650)
            self.assertTrue(all(path.read_bytes().startswith(b"%PDF-") for path in pdf_files))
            statistics = pd.read_csv(workspace / "output/estadistica/prevalencias_crudas.csv").set_index("Variable")
            self.assertEqual(statistics.loc["ASI", "Prevalencia_ponderada (%)"], 50)
            self.assertEqual(statistics.loc["Intento de suicidio", "Prevalencia_ponderada (%)"], 60)
            self.assertEqual(statistics.loc["Ideación suicida", "Denominador (n)"], 3)
        pd.testing.assert_frame_equal(data, original)
        self.assertEqual(plt.rcParams["font.size"], original_font_size)
        self.assertFalse(plt.get_fignums())


def model_plot_data():
    names = list(MODEL_COLORS)
    comparison = pd.DataFrame({"modelo": names, "MCC_media": [0.3, 0.2, 0.1, 0.05], "MCC_sd": [0.01] * 4})
    predictions = pd.DataFrame({"y": [0, 1, 0, 1]})
    for name in names:
        predictions[f"{name}_pred"] = [0, 1, 1, 0]
        predictions[f"{name}_proba"] = [0.1, 0.9, 0.6, 0.3]
    curve = pd.DataFrame(
        {"n_features": [3, 4, 5], "mean_test_score": [0.1, 0.2, 0.15], "std_test_score": [0.01, 0.02, 0.03]}
    )
    return comparison, predictions, curve


class PipelineFigureStyleTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")

    def test_target_distribution_restores_original_colors_and_counts(self):
        with patch("suicide_predictors.plots.save") as save:
            plot_target_distribution(pd.Series([0, 0, 0, 1]), Path("unused.png"))
        axes = save.call_args.args[0].axes[0]
        self.assertEqual([bar.get_height() for bar in axes.patches], [3, 1])
        self.assertEqual([to_hex(bar.get_facecolor()) for bar in axes.patches], [PALETTE["blue"], PALETTE["red"]])
        self.assertEqual([text.get_text() for text in axes.texts], ["3\n(75,0 %)", "1\n(25,0 %)"])

    def test_confusion_matrices_restore_blues_without_changing_counts(self):
        comparison, predictions, _ = model_plot_data()
        with patch("suicide_predictors.plots.save") as save:
            plot_confusion_matrices(comparison, predictions, Path("unused"))
        for axes in save.call_args.args[0].axes:
            self.assertEqual(axes.images[0].get_cmap().name, "Blues")
            np.testing.assert_equal(axes.images[0].get_array(), np.ones((2, 2)))
            self.assertIn("MCC CV =", axes.get_title())

    def test_mcc_restores_mauve_and_order_without_changing_scores(self):
        comparison, _, _ = model_plot_data()
        original = comparison.copy(deep=True)
        with patch("suicide_predictors.plots.save") as save:
            plot_model_comparison(comparison, Path("unused"))
        axes = save.call_args.args[0].axes[0]
        np.testing.assert_equal([bar.get_width() for bar in axes.patches], [0.05, 0.1, 0.2, 0.3])
        self.assertTrue(all(to_hex(bar.get_facecolor()) == PALETTE["mauve"] for bar in axes.patches))
        pd.testing.assert_frame_equal(comparison, original)

    def test_roc_restores_each_models_original_color(self):
        from sklearn.metrics import roc_curve

        comparison, predictions, _ = model_plot_data()
        with patch("suicide_predictors.plots.save") as save:
            plot_roc_curves(comparison, predictions, Path("unused"))
        axes = save.call_args.args[0].axes[0]
        for line, name in zip(axes.lines[:4], comparison["modelo"], strict=True):
            fpr, tpr, _ = roc_curve(predictions["y"], predictions[f"{name}_proba"])
            self.assertEqual(line.get_color(), MODEL_COLORS[name])
            np.testing.assert_equal(line.get_xdata(), fpr)
            np.testing.assert_equal(line.get_ydata(), tpr)
            self.assertIn("AUC agrupado", line.get_label())

    def test_selection_restores_curve_band_and_reference_colors(self):
        _, _, curve = model_plot_data()
        with patch("suicide_predictors.plots.save") as save:
            plot_selection_curve(curve, 4, Path("unused.png"))
        axes = save.call_args.args[0].axes[0]
        self.assertEqual(axes.lines[0].get_color(), SELECTION_COLORS["curve"])
        self.assertEqual(axes.lines[1].get_color(), SELECTION_COLORS["selected"])
        self.assertEqual(to_hex(axes.collections[0].get_facecolor()[0]), SELECTION_COLORS["band"].lower())
        np.testing.assert_equal(axes.lines[0].get_ydata(), curve["mean_test_score"])
        np.testing.assert_equal(axes.lines[1].get_xdata(), [4, 4])

    def test_shap_importance_restores_original_blue(self):
        from suicide_predictors.plots import bar

        importance = pd.Series([0.1, 0.03], index=["asi", "edad"])
        with patch("suicide_predictors.plots.save") as save:
            bar(importance, "SHAP", "Media |SHAP|", Path("unused.png"))
        axes = save.call_args.args[0].axes[0]
        self.assertTrue(all(to_hex(patch.get_facecolor()) == PALETTE["blue"] for patch in axes.patches))
        np.testing.assert_equal([patch.get_width() for patch in axes.patches], [0.03, 0.1])

    def test_remaining_figures_export_png_pdf_without_retraining_or_altering_shap(self):
        import shap

        comparison, predictions, curve = model_plot_data()
        values = np.array([[0.1, -0.1], [0.02, 0.03], [-0.1, 0.2], [0.1, -0.03]])
        data = np.array([[0, 10], [1, 12], [0, 15], [1, 18]])
        explanation = shap.Explanation(
            values=values.copy(), data=data.copy(), base_values=np.zeros(4), feature_names=["asi", "edad"]
        )
        importance = pd.Series(np.abs(values).mean(axis=0), index=explanation.feature_names)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            classification_figures(comparison, predictions, curve, 4, workspace)
            plot_target_distribution(
                pd.Series([0, 0, 0, 1]), workspace / "figuras/preprocesamiento/14_target_intento_frecuencias.png"
            )
            shap_figures(importance, explanation, workspace / "figuras/shap", model_name="Regresión Logística")
            self.assertEqual(len(list((workspace / "figuras").rglob("*.png"))), 7)
            self.assertEqual(len(list((workspace / "figuras").rglob("*.pdf"))), 7)
        np.testing.assert_array_equal(explanation.values, values)
        np.testing.assert_array_equal(explanation.data, data)
        self.assertFalse(plt.get_fignums())


if __name__ == "__main__":
    unittest.main()
