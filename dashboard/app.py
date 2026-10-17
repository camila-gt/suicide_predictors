import sys
from pathlib import Path

# Agrega la raiz del proyecto al path para imports del dashboard
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from dashboard.data import artifact_root, begin_snapshot, load_all, load_mejor_modelo_info
from dashboard.tabs import inicio, mapa, metodos, modelos_ml, muestra, prediccion, shap_interpretabilidad
from suicide_predictors import SCHEMA_VERSION

st.set_page_config(
    page_title="Clasificación de antecedente de intento – ENSANUT 2024",
    page_icon="🔬",
    layout="wide",
)

begin_snapshot()
df, ml_df, geojson = load_all()
if not (artifact_root() / "manifest.json").exists():
    st.warning(
        "Aún no hay una ejecución publicada del pipeline actual. Los artefactos existentes son solo de consulta."
    )
elif load_mejor_modelo_info().get("schema_version") != SCHEMA_VERSION:
    st.warning(
        "Los resultados visibles usan un esquema anterior. Ejecuta el pipeline para generar los resultados actuales."
    )

# Sidebar
with st.sidebar:
    st.markdown(
        "<h2 style='text-align: center; margin-top: 0px; font-weight: 700;'>🔬 Clasificación de intento</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Antecedente de intento · ENSANUT 2024")
    st.divider()

    # Menú de navegación
    nav_option = st.radio(
        "Navegación",
        [
            "🏠 Inicio",
            "📊 Muestra total",
            "💊 Métodos de intento",
            "⚙️ Modelos de ML",
            "🐝 Interpretabilidad SHAP",
            "🔮 Predicción interactiva",
            "🗺️ Mapa geográfico",
        ],
        key="navigation",
    )

css_path = Path(__file__).with_name("styles.css")
css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)

routes = {
    "🏠 Inicio": lambda: inicio.render(df),
    "📊 Muestra total": lambda: muestra.render(df),
    "💊 Métodos de intento": lambda: metodos.render(df),
    "⚙️ Modelos de ML": lambda: modelos_ml.render(),
    "🐝 Interpretabilidad SHAP": lambda: shap_interpretabilidad.render(),
    "🔮 Predicción interactiva": lambda: prediccion.render(),
    "🗺️ Mapa geográfico": lambda: mapa.render(df, geojson),
}

routes.get(nav_option, routes["🏠 Inicio"])()

# Pie
st.divider()
st.caption(
    "Fuente: ENSANUT Continua 2024 - INSP / SSA - "
    "Módulo de Minería de Datos (KDD) e Inteligencia Artificial Explicable (SHAP) - "
    "Los datos mostrados no son para diagnóstico clínico."
)
