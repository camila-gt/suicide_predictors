import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from joblib import dump, load

from suicide_predictors.__main__ import main
from suicide_predictors.artifacts import active_workspace, publish, write_json
from suicide_predictors.estimators import ApplicabilityImputer, SurveyClassifier, balanced_weights, prediction_pipeline
from suicide_predictors.explain import explain
from suicide_predictors.inference import prediction_frame
from suicide_predictors.modeling import (
    MODEL_NAMES,
    ExperimentConfig,
    ModelData,
    checked_splits,
    select_features,
    selection_frequency,
    summarize_scores,
    train,
    validate_ml,
)
from suicide_predictors.preparation import any_positive, build_ml, prepare
from suicide_predictors.schema import FEATURES


def raw_row(**changes):
    values = {
        "edad": 14,
        "sexo": 1,
        "ponde_f": "2,5",
        "upm": "UPM001",
        "estrato": 1,
        "d0810": 3,
        "d0819": 3,
        "d0101": 3,
        "d0107": 4,
        "d0108": 5,
        "d0109": np.nan,
        "d0110": np.nan,
        "d0111": np.nan,
        "d0112": np.nan,
        "d0403": 2,
        "d0406": np.nan,
        "d0407": np.nan,
        "d0410": 1,
        "d0411": 1,
        "d0412": 1,
        "d0425": np.nan,
        "d0427": np.nan,
        "d0421": 3,
        "d0415": 1,
        "d0416": 1,
        "d0417": 1,
        "d0418": 1,
        "d0419": 1,
        "d0420": 1,
        "d0305": 0,
        "d0309": np.nan,
        "d0313": np.nan,
        "d0801": 2,
    }
    values.update({f"d0601{c}": 2 for c in "abcdefg"})
    values.update({f"d06a{i}": 1 for i in range(1, 11)})
    values.update({f"d0901{c}": 2 for c in "abcdefghijk"})
    values.update({f"d01a08{c}": 2 for c in "abcdefghi"})
    values.update({f"d01a03{c}": 2 for c in "abcd"})
    values.update({f"d01a04{c}": np.nan for c in "abcd"})
    values.update({f"d02o1{c}": 2 for c in "abcd"})
    values.update(changes)
    return values


def synthetic_frame(n=80):
    rng = np.random.default_rng(2)
    feature_columns = {}
    for name, spec in FEATURES.items():
        values = rng.integers(spec.minimum, spec.maximum + 1, n)
        feature_columns[name] = values.astype(float)
    frame = pd.DataFrame(feature_columns)
    frame["row_id"] = np.arange(n)
    frame["upm"] = np.repeat(np.arange(n // 2), 2).astype(str)
    frame["estrato"] = 1
    frame["ponde_f"] = rng.uniform(1, 4, n)
    frame["intento_bin"] = np.tile([0, 1], n // 2)
    frame.loc[::9, "depresion_cesd"] = np.nan
    return frame


class PreparationTests(unittest.TestCase):
    def test_prepare_generates_original_target_figure(self):
        descriptive_fields = {
            "d0811": np.nan,
            "d0812": np.nan,
            "d0813": np.nan,
            "d0814": np.nan,
            "d0815": np.nan,
            "d0817": 2,
            "entidad": 1,
        }
        rows = [raw_row(**descriptive_fields), raw_row(d0819=1, **descriptive_fields)]
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            raw_path = workspace / "raw.csv"
            pd.DataFrame(rows).to_csv(raw_path, sep=";", index=False, encoding="utf-8-sig")
            _, ml = prepare(raw_path, workspace)
            self.assertEqual(ml.intento_bin.tolist(), [0, 1])
            figure = workspace / "figuras/preprocesamiento/14_target_intento_frecuencias.png"
            self.assertTrue(figure.is_file())
            self.assertTrue(figure.with_suffix(".pdf").is_file())

    def test_partial_negative_stays_unknown(self):
        result = any_positive(pd.DataFrame([[2, np.nan], [1, np.nan], [2, 2]]))
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertEqual(result.iloc[1:].tolist(), [1, 0])

    def test_nonviolent_discipline_is_not_violence(self):
        result = build_ml(pd.DataFrame([raw_row(d0901b=1, d0901e=1)]))
        self.assertEqual(result.loc[0, "violencia_familiar"], 0)

    def test_walking_without_assistance_is_used(self):
        result = build_ml(pd.DataFrame([raw_row(d0411=3)]))
        self.assertEqual(result.loc[0, "discapacidad_fisica"], 1)

    def test_medication_use_as_prescribed_is_not_nonprescribed(self):
        for group in "abcd":
            with self.subTest(group=group):
                row = raw_row(**{f"d01a03{group}": 1, f"d01a04{group}": 2})
                result = build_ml(pd.DataFrame([row]))
                self.assertEqual(result.loc[0, "sustancias_medicas"], 0)

    def test_medication_without_or_against_prescription_is_positive(self):
        for group in "abcd":
            with self.subTest(group=group):
                row = raw_row(**{f"d01a03{group}": 1, f"d01a04{group}": 1})
                result = build_ml(pd.DataFrame([row]))
                self.assertEqual(result.loc[0, "sustancias_medicas"], 1)

    def test_medication_mode_missing_stays_unknown(self):
        for missing in (np.nan, 9):
            with self.subTest(missing=missing):
                result = build_ml(pd.DataFrame([raw_row(d01a03a=1, d01a04a=missing)]))
                self.assertTrue(np.isnan(result.loc[0, "sustancias_medicas"]))

    def test_medication_unknown_use_stays_unknown(self):
        result = build_ml(pd.DataFrame([raw_row(d01a03a=9)]))
        self.assertTrue(np.isnan(result.loc[0, "sustancias_medicas"]))

    def test_medication_positive_overrides_other_unknown_groups(self):
        row = raw_row(d01a03a=1, d01a04a=1, d01a03b=1, d01a04b=9)
        result = build_ml(pd.DataFrame([row]))
        self.assertEqual(result.loc[0, "sustancias_medicas"], 1)

    def test_no_medication_use_does_not_require_mode_responses(self):
        result = build_ml(pd.DataFrame([raw_row()]))
        self.assertEqual(result.loc[0, "sustancias_medicas"], 0)

    def test_medication_mode_columns_are_required(self):
        raw = pd.DataFrame([raw_row()]).drop(columns="d01a04a")
        with self.assertRaisesRegex(ValueError, "d01a04a"):
            build_ml(raw)

    def test_substances_do_not_apply_before_age_12_even_with_responses(self):
        for age in (10, 11, 12, 19):
            with self.subTest(age=age):
                row = raw_row(edad=age, d01a08a=1, d01a03a=1, d01a04a=1)
                result = build_ml(pd.DataFrame([row]))
                expected = -1 if age < 12 else 1
                for feature in ("drogas_ilegales", "sustancias_medicas"):
                    self.assertEqual(result.loc[0, feature], expected)

    def test_eligible_missing_substances_remain_unknown(self):
        changes = {f"d01a08{group}": np.nan for group in "abcdefghi"}
        changes.update({f"d01a03{group}": np.nan for group in "abcd"})
        result = build_ml(pd.DataFrame([raw_row(edad=12, **changes)]))
        self.assertTrue(result[["drogas_ilegales", "sustancias_medicas"]].isna().all().all())

    def test_annual_binge_skip_deduces_zero_for_each_sex(self):
        rows = [
            raw_row(sexo=1, d0108=1, d0109=6, d0110=1, d0111=np.nan, d0112=1),
            raw_row(sexo=2, d0108=1, d0109=1, d0110=6, d0111=1, d0112=np.nan),
        ]
        result = build_ml(pd.DataFrame(rows))
        self.assertEqual(result["binge_alcohol"].tolist(), [0, 0])

    def test_annual_binge_does_not_determine_monthly_binge(self):
        for frequency in (1, 2, 3, 4, 5, 9, np.nan):
            with self.subTest(frequency=frequency):
                row = raw_row(d0108=1, d0109=frequency, d0110=6)
                result = build_ml(pd.DataFrame([row]))
                self.assertTrue(np.isnan(result.loc[0, "binge_alcohol"]))

    def test_explicit_monthly_binge_responses_are_used(self):
        rows = [
            raw_row(d0108=1, d0109=3, d0111=1),
            raw_row(d0108=1, d0109=3, d0111=2),
            raw_row(d0108=1, d0109=3, d0111=9),
        ]
        result = build_ml(pd.DataFrame(rows))
        self.assertEqual(result["binge_alcohol"].iloc[:2].tolist(), [1, 0])
        self.assertTrue(np.isnan(result.loc[2, "binge_alcohol"]))

    def test_no_alcohol_in_year_deduces_no_monthly_binge(self):
        for response in (5, 6):
            with self.subTest(response=response):
                result = build_ml(pd.DataFrame([raw_row(d0108=response)]))
                self.assertEqual(result.loc[0, "binge_alcohol"], 0)

    def test_severe_100m_walking_recovers_skipped_500m_response(self):
        for assistance in (1, 2):
            for severity in (3, 4):
                with self.subTest(assistance=assistance, severity=severity):
                    row = raw_row(d0403=assistance, d0407=np.nan, d0411=np.nan)
                    if assistance == 1:
                        row["d0406"] = severity
                    else:
                        row["d0410"] = severity
                    result = build_ml(pd.DataFrame([row]))
                    self.assertEqual(result.loc[0, "discapacidad_fisica"], 1)

    def test_severity_from_other_walking_route_is_ignored(self):
        rows = [
            raw_row(d0403=1, d0406=2, d0407=2, d0410=4, d0411=4),
            raw_row(d0403=2, d0406=4, d0407=4, d0410=1, d0411=1),
        ]
        result = build_ml(pd.DataFrame(rows))
        self.assertEqual(result["discapacidad_fisica"].tolist(), [0, 0])

    def test_nonsevere_100m_does_not_resolve_missing_500m(self):
        result = build_ml(pd.DataFrame([raw_row(d0410=2, d0411=np.nan)]))
        self.assertTrue(np.isnan(result.loc[0, "discapacidad_fisica"]))

    def test_child_walking_answers_do_not_replace_adult_questions(self):
        row = raw_row(edad=19, d0410=4, d0411=4, d0425=1, d0427=1)
        result = build_ml(pd.DataFrame([row]))
        self.assertEqual(result.loc[0, "discapacidad_fisica"], 0)

    def test_structural_absence_and_unknown_are_distinct(self):
        frame = pd.DataFrame([raw_row(edad=19, d0421=np.nan), raw_row(d0421=9)])
        result = build_ml(frame)
        self.assertEqual(result.loc[0, "ansiedad_frec"], -1)
        self.assertTrue(np.isnan(result.loc[1, "ansiedad_frec"]))
        self.assertEqual(result.loc[0, "embarazo"], -1)

    def test_invalid_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "d0819"):
            build_ml(pd.DataFrame([raw_row(d0819=9)]))

    def test_invalid_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positivos"):
            build_ml(pd.DataFrame([raw_row(ponde_f=0)]))

    def test_adult_mobility_uses_adult_questions(self):
        result = build_ml(pd.DataFrame([raw_row(edad=19, d0425=3, d0427=1)]))
        self.assertEqual(result.loc[0, "discapacidad_fisica"], 1)

    def test_vape_unfamiliar_is_not_current_use(self):
        self.assertEqual(build_ml(pd.DataFrame([raw_row()])).loc[0, "vapea_actual"], 0)


class ModelTests(unittest.TestCase):
    def test_median_excludes_structural_values(self):
        X = np.array([[-1], [-1], [2], [4], [np.nan]])
        imputer = ApplicabilityImputer().fit(X)
        self.assertEqual(imputer.statistics_[0], 3)
        transformed = imputer.transform(X)
        np.testing.assert_equal(transformed[:, 0], [3, 3, 2, 4, 3])
        np.testing.assert_equal(transformed[:, 1], [1, 1, 0, 0, 0])
        np.testing.assert_equal(transformed[:, 2], [0, 0, 0, 0, 1])

    def test_empty_feature_is_not_dropped(self):
        X = np.array([[np.nan, 1], [np.nan, 2]])
        self.assertEqual(ApplicabilityImputer().fit_transform(X).shape, (2, 6))

    def test_structural_sentinel_cannot_replace_asi(self):
        frame = synthetic_frame()
        frame.loc[0, "asi"] = -1
        with self.assertRaisesRegex(ValueError, "asi"):
            validate_ml(frame)

    def test_ml_validator_accepts_structural_substance_values(self):
        frame = synthetic_frame()
        frame.loc[0, "edad"] = 11
        frame.loc[0, ["drogas_ilegales", "sustancias_medicas"]] = -1
        validate_ml(frame)

    def test_balance_uses_training_weighted_mass(self):
        weights = balanced_weights([0, 0, 0, 1], [1, 1, 1, 100])
        self.assertAlmostEqual(weights[:3].sum(), weights[3])

    def test_partitions_are_disjoint(self):
        frame = synthetic_frame()
        for training, validation in checked_splits(
            frame[list(FEATURES)], frame.intento_bin, frame.upm, folds=3, seed=42
        ):
            self.assertFalse(set(frame.upm.iloc[training]) & set(frame.upm.iloc[validation]))

    def test_rfecv_fits_imputer_on_inner_training(self):
        frame = synthetic_frame()
        sizes = []
        original = ApplicabilityImputer.fit

        def recorded(imputer, X, y=None):
            sizes.append(len(X))
            return original(imputer, X, y)

        config = ExperimentConfig(jobs=1, min_features=2)
        with patch.object(ApplicabilityImputer, "fit", recorded):
            select_features(
                frame[list(FEATURES)[:3]], frame.intento_bin, frame.upm, frame.ponde_f, folds=2, trees=3, config=config
            )
        self.assertIn(40, sizes)
        self.assertIn(80, sizes)

    def test_parallel_selection_uses_importable_estimators(self):
        frame = synthetic_frame()
        selector, features = select_features(
            frame[list(FEATURES)[:3]],
            frame.intento_bin,
            frame.upm,
            frame.ponde_f,
            folds=2,
            trees=3,
            config=ExperimentConfig(jobs=2, min_features=3),
        )
        self.assertEqual(selector.n_features_, len(features))

    def test_persistence_includes_preprocessing(self):
        frame = synthetic_frame()
        features = ["edad", "depresion_cesd", "asi"]
        for kind in ("Regresión Logística", "Random Forest", "Gradient Boosting", "SVM (RBF)"):
            with self.subTest(kind=kind):
                model = prediction_pipeline(features, kind, trees=3).fit(
                    frame[features],
                    frame.intento_bin,
                    model__sample_weight=frame.ponde_f,
                )
                stream = io.BytesIO()
                dump(model, stream)
                stream.seek(0)
                restored = load(stream)
                np.testing.assert_allclose(
                    model.predict_proba(frame[features]), restored.predict_proba(frame[features])
                )
                self.assertIsInstance(restored.named_steps["model"], SurveyClassifier)


class ModelingHelperTests(unittest.TestCase):
    def test_subset_keeps_features_labels_and_weights_aligned(self):
        frame = synthetic_frame()
        positions = np.array([7, 2, 11])
        subset = ModelData.from_frame(frame).take(positions)
        pd.testing.assert_frame_equal(subset.features, frame[list(FEATURES)].iloc[positions])
        pd.testing.assert_series_equal(subset.target, frame.intento_bin.iloc[positions])
        pd.testing.assert_series_equal(subset.groups, frame.upm.iloc[positions])
        pd.testing.assert_series_equal(subset.weights, frame.ponde_f.iloc[positions])
        pd.testing.assert_series_equal(subset.row_ids, frame.row_id.iloc[positions])

    def test_selection_frequency_counts_folds(self):
        selections = [{"features": ["edad", "asi"]}, {"features": ["asi"]}]
        result = selection_frequency(selections).set_index("feature")
        self.assertEqual(result.loc["edad", "selection_frequency"], 0.5)
        self.assertEqual(result.loc["asi", "selection_frequency"], 1.0)
        self.assertEqual(result.loc["fuma_actual", "selection_frequency"], 0.0)

    def test_score_summary_preserves_tie_order(self):
        means = {
            "Regresión Logística": 0.3,
            "Random Forest": 0.2,
            "Gradient Boosting": 0.3,
            "SVM (RBF)": 0.1,
        }
        scores = []
        for name in MODEL_NAMES:
            scores.append({"modelo": name, "fold": 0, "MCC": means[name]})
        comparison = summarize_scores(pd.DataFrame(scores))
        self.assertEqual(
            comparison["modelo"].tolist(),
            [
                "Regresión Logística",
                "Gradient Boosting",
                "Random Forest",
                "SVM (RBF)",
            ],
        )
        self.assertTrue(comparison["MCC_sd"].eq(0).all())


class InferenceTests(unittest.TestCase):
    def test_missing_widget_is_not_silently_zero(self):
        with self.assertRaisesRegex(ValueError, "faltantes"):
            prediction_frame(["fuma_actual"], {}, age=14, female=False)

    def test_fractional_binary_input_is_rejected(self):
        with self.assertRaises(ValueError):
            prediction_frame(["fuma_actual"], {"fuma_actual": 0.5}, age=14, female=False)

    def test_unknown_asi_is_not_imputed(self):
        with self.assertRaisesRegex(ValueError, "ASI no se imputa"):
            prediction_frame(["asi"], {"asi": np.nan}, age=14, female=False)

    def test_unknown_is_preserved(self):
        frame = prediction_frame(["fuma_actual"], {"fuma_actual": np.nan}, age=14, female=False)
        self.assertTrue(frame.isna().all().all())

    def test_substance_inference_respects_minimum_age(self):
        features = ["drogas_ilegales", "sustancias_medicas"]
        for age in (10, 11, 12, 19):
            with self.subTest(age=age):
                if age < 12:
                    expected = -1.0
                    invalid = 0.0
                else:
                    expected = 0.0
                    invalid = -1.0
                values = dict.fromkeys(features, expected)
                frame = prediction_frame(features, values, age=age, female=False)
                self.assertTrue(frame.eq(expected).all().all())
                with self.assertRaises(ValueError):
                    prediction_frame(features, dict.fromkeys(features, invalid), age=age, female=False)

    def test_structural_value_requires_ineligible_profile(self):
        frame = prediction_frame(["ansiedad_frec"], {"ansiedad_frec": -1}, age=19, female=True)
        self.assertEqual(frame.iloc[0, 0], -1)
        with self.assertRaises(ValueError):
            prediction_frame(["ansiedad_frec"], {"ansiedad_frec": -1}, age=14, female=True)


class ArtifactTests(unittest.TestCase):
    def test_failed_run_cannot_replace_active_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good, bad = root / "output/runs/good", root / "output/runs/bad"
            write_json(good / "manifest.json", {"status": "complete"})
            write_json(bad / "manifest.json", {"status": "failed"})
            publish(good, root)
            with self.assertRaises(ValueError):
                publish(bad, root)
            self.assertEqual(active_workspace(root), good)

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            write_json(workspace / "manifest.json", {"status": "complete"})
            with self.assertRaisesRegex(ValueError, "ya existe"):
                main(["explain", "--workspace", str(workspace), "--raw", "does-not-exist.csv"])
            self.assertEqual(json.loads((workspace / "manifest.json").read_text())["status"], "complete")

    def test_small_experiment_writes_complete_artifacts(self):
        frame = synthetic_frame()
        config = ExperimentConfig(
            outer_folds=2,
            inner_folds=2,
            final_folds=2,
            selection_trees=3,
            final_selection_trees=3,
            forest_trees=3,
            min_features=20,
            jobs=1,
            bootstrap_repetitions=3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "data").mkdir()
            frame.to_csv(workspace / "data/adolescentes_ml.csv", index=False)
            info = train(frame, workspace, config)
            directory = workspace / "output/clasificacion"
            self.assertEqual(info["n_dev"] + info["n_test"], len(frame))
            self.assertTrue((directory / "modelo_evaluado.joblib").is_file())
            self.assertTrue((directory / "predicciones_oof.csv").is_file())
            json.loads((directory / "mejor_modelo_info.json").read_text(encoding="utf-8"))
            model = load(directory / "mejor_modelo.joblib")
            self.assertEqual(model.predict_proba(frame[info["features"]]).shape, (len(frame), 2))
            explanation = explain(workspace, sample_size=6, background_size=5, permutations=1)
            self.assertEqual(explanation["output"], "positive_class_probability")
            self.assertTrue((workspace / "output/shap/explicaciones.npz").exists())


if __name__ == "__main__":
    unittest.main()
