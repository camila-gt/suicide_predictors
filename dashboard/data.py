import json
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline

from suicide_predictors.artifacts import PROJECT_ROOT, active_workspace

ROOT = PROJECT_ROOT


def begin_snapshot() -> None:
    st.session_state["_artifact_workspace"] = str(active_workspace(ROOT))


def artifact_root() -> Path:
    snapshot = st.session_state.get("_artifact_workspace")
    return Path(snapshot) if snapshot is not None else active_workspace(ROOT)


def signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@st.cache_data
def _csv(path: str, modified: int, size: int) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def _json(path: str, modified: int, size: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource
def _model(path: str, modified: int, size: int) -> Pipeline:
    return load(path)


def load_df() -> pd.DataFrame:
    frame = _csv(*signature(artifact_root() / "preprocesamiento" / "adolescentes_limpio.csv"))
    frame["grupo_edad"] = pd.Categorical(frame["grupo_edad"], categories=["10-12", "13-15", "16-19"], ordered=True)
    return frame


def load_ml_df() -> pd.DataFrame:
    return _csv(*signature(artifact_root() / "preprocesamiento" / "adolescentes_ml.csv"))


def load_comparacion_modelos(mode: str = "bin") -> pd.DataFrame:
    path = artifact_root() / "clasificacion" / "comparacion_modelos.csv"
    return _csv(*signature(path)) if path.exists() else pd.DataFrame()


def load_shap_importancia(mode: str = "bin") -> pd.DataFrame:
    path = artifact_root() / "shap" / "shap_importancia.csv"
    return _csv(*signature(path)) if path.exists() else pd.DataFrame()


def load_mejor_modelo_info(mode: str = "bin") -> dict:
    path = artifact_root() / "clasificacion" / "mejor_modelo_info.json"
    return _json(*signature(path)) if path.exists() else {}


def load_mejor_modelo(mode: str = "bin") -> Pipeline | None:
    path = artifact_root() / "clasificacion" / "mejor_modelo.joblib"
    return _model(*signature(path)) if path.exists() else None


def load_geojson() -> dict:
    return _json(*signature(ROOT / "data" / "states_simple.geojson"))


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    with st.spinner("Cargando datos y modelos..."):
        return load_df(), load_ml_df(), load_geojson()
