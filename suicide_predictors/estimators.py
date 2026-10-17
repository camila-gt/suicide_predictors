from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from suicide_predictors.schema import FEATURE_CHANNELS


class ApplicabilityImputer(TransformerMixin, BaseEstimator):
    """Codifica valores, no aplicabilidad y desconocimiento como canales separados."""

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> Self:
        values = np.asarray(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        observed_values = np.where(values == -1, np.nan, values)
        self.imputer_.fit(observed_values)
        self.statistics_ = self.imputer_.statistics_
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "imputer_")
        values = np.asarray(X, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("La entrada no coincide con las columnas ajustadas")
        structural = values == -1
        unknown = np.isnan(values)
        observed_values = np.where(structural, np.nan, values)
        transformed = self.imputer_.transform(observed_values)
        structural_indicator = structural.astype(float)
        unknown_indicator = unknown.astype(float)
        return np.concatenate([transformed, structural_indicator, unknown_indicator], axis=1)


def balanced_weights(y: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    labels = np.asarray(y)
    if sample_weight is None:
        weights = np.ones(len(labels))
    else:
        weights = np.asarray(sample_weight, dtype=float)
    matching_shape = weights.shape == labels.shape
    finite_weights = np.isfinite(weights).all()
    positive_weights = (weights > 0).all()
    if not matching_shape or not finite_weights or not positive_weights:
        raise ValueError("Los pesos deben ser finitos, positivos y coincidir con las etiquetas")
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("El entrenamiento requiere ambas clases")
    balanced = weights / weights.mean()
    total_weight = weights.sum()
    for label in classes:
        belongs_to_class = labels == label
        class_weight = weights[belongs_to_class].sum()
        class_balance_factor = total_weight / (len(classes) * class_weight)
        balanced[belongs_to_class] *= class_balance_factor
    return balanced


class SurveyClassifier(ClassifierMixin, BaseEstimator):
    """Ajusta imputación, escalamiento y balanceo solo con el entrenamiento recibido."""

    def __init__(self, kind: str = "Random Forest", n_estimators: int = 100, random_state: int = 42, n_jobs: int = 1):
        self.kind = kind
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> Self:
        self.n_features_in_ = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        steps = [("imputer", ApplicabilityImputer())]
        if self.kind in ("Regresión Logística", "SVM (RBF)"):
            steps.append(("scaler", StandardScaler()))
        self.preprocessor_ = Pipeline(steps)
        prepared = self.preprocessor_.fit_transform(X)
        self.classifier_ = self._create_classifier()
        training_weights = balanced_weights(y, sample_weight)
        self.classifier_.fit(prepared, y, sample_weight=training_weights)
        self.classes_ = self.classifier_.classes_
        return self

    def _create_classifier(self):
        if self.kind == "Regresión Logística":
            return LogisticRegression(max_iter=5000, random_state=self.random_state)
        if self.kind == "Random Forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        if self.kind == "Gradient Boosting":
            return GradientBoostingClassifier(random_state=self.random_state)
        if self.kind == "SVM (RBF)":
            return SVC(kernel="rbf", probability=True, random_state=self.random_state)
        raise ValueError(f"Clasificador desconocido: {self.kind}")

    @property
    def feature_importances_(self) -> np.ndarray:
        check_is_fitted(self, "classifier_")
        importances = self.classifier_.feature_importances_
        channel_importances = np.split(importances, len(FEATURE_CHANNELS))
        value_importance, structural_importance, unknown_importance = channel_importances
        return np.sum([value_importance, structural_importance, unknown_importance], axis=0)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "classifier_")
        prepared = self.preprocessor_.transform(X)
        return self.classifier_.predict(prepared)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "classifier_")
        prepared = self.preprocessor_.transform(X)
        return self.classifier_.predict_proba(prepared)


class FeatureColumns(TransformerMixin, BaseEstimator):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> Self:
        self.transform(X)
        self.n_features_in_ = len(self.columns)
        self.feature_names_in_ = np.asarray(self.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"Faltan predictores: {sorted(missing)}")
        return X.loc[:, list(self.columns)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.columns, dtype=object)


def prediction_pipeline(features: list[str], kind: str, *, trees: int = 400, seed: int = 42, jobs: int = 1) -> Pipeline:
    return Pipeline(
        [
            ("columns", FeatureColumns(features)),
            ("model", SurveyClassifier(kind=kind, n_estimators=trees, random_state=seed, n_jobs=jobs)),
        ]
    )
