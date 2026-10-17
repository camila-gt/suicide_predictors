import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from dashboard.data import load_mejor_modelo, load_mejor_modelo_info
from dashboard.presentation import render_table
from suicide_predictors import SCHEMA_VERSION
from suicide_predictors.inference import prediction_frame
from suicide_predictors.schema import ANXIETY_OPTIONS, FEATURES, NOT_APPLICABLE, is_applicable


def read_feature(
    name: str,
    *,
    age: int,
    female: bool,
    sexual_history: bool | None,
    entered_values: dict[str, float],
) -> float:
    spec = FEATURES[name]
    if not is_applicable(name, age, female, sexual_history):
        st.caption(f"{spec.label}: no aplica según el cuestionario.")
        return NOT_APPLICABLE
    if spec.sexual_history and sexual_history is None:
        st.caption(f"{spec.label}: información desconocida.")
        return np.nan
    if name == "binge_alcohol" and entered_values.get("alcohol_12m") == 0:
        st.caption(f"{spec.label}: no, porque no hubo consumo en los últimos 12 meses.")
        return 0.0
    if spec.maximum == 1:
        choice = st.selectbox(spec.label, ["Sin información", "No", "Sí"], key=f"feature_{name}")
        response_codes = {"Sin información": np.nan, "No": 0.0, "Sí": 1.0}
        return response_codes[choice]
    if name == "ansiedad_frec":
        options = ["Sin información", *ANXIETY_OPTIONS.values()]
        choice = st.selectbox(spec.label, options, key=f"feature_{name}")
        for code, label in ANXIETY_OPTIONS.items():
            if label == choice:
                return float(code)
        return np.nan
    if st.checkbox(f"Sin información: {spec.label}", key=f"missing_{name}"):
        return np.nan
    value = st.slider(spec.label, spec.minimum, spec.maximum, spec.default, key=f"feature_{name}")
    return float(value)


def read_profile(features: list[str]) -> tuple[int, bool, bool | None, dict[str, float]]:
    age = st.slider("Edad (años)", 10, 19, 14)
    female = st.selectbox("Sexo registrado en la encuesta", ["Hombre", "Mujer"]) == "Mujer"
    sexual_history = None
    needs_sexual_history = any(FEATURES[name].sexual_history for name in features)
    if age >= 12 and needs_sexual_history:
        options = ["Sin información", "No", "Sí"]
        choice = st.selectbox("Inicio de vida sexual (para aplicar el cuestionario)", options)
        response_codes = {"Sin información": None, "No": False, "Sí": True}
        sexual_history = response_codes[choice]
    values = {"edad": float(age), "sexo_Mujer": float(female)}
    columns = st.columns(2)
    field_index = 0
    for name in features:
        if name in values:
            continue
        column = columns[field_index % len(columns)]
        with column:
            values[name] = read_feature(
                name,
                age=age,
                female=female,
                sexual_history=sexual_history,
                entered_values=values,
            )
        field_index += 1
    return age, female, sexual_history, values


def display_value(value: float) -> str:
    if value == NOT_APPLICABLE:
        return "No aplica"
    if pd.isna(value):
        return "Desconocido"
    return str(value)


def show_prediction(frame: pd.DataFrame, model: Pipeline) -> None:
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(model.predict(frame)[0])
    st.metric("Salida del modelo para la clase «Intentó»", f"{100 * probability:.1f}%")
    if prediction:
        label = "con antecedente de intento"
    else:
        label = "sin antecedente de intento"
    st.write(f"Clase estimada: {label}.")
    st.caption(
        "Esta salida no es una probabilidad clínica calibrada. El modelo se ajustó con balanceo de clases "
        "y ponderadores muestrales; no se asignan categorías clínicas de riesgo."
    )
    display = frame.T.rename(columns={0: "Valor"})
    display.index = [FEATURES[name].label for name in frame.columns]
    display["Valor"] = display["Valor"].map(display_value)
    render_table(display.rename_axis("Predictor").reset_index())
    st.caption(
        "Los valores desconocidos se procesan con el imputador guardado en el modelo; no se sustituyen por cero en el formulario."
    )


def render() -> None:
    st.subheader("Simulador académico de clasificación")
    st.info(
        "El modelo distingue el antecedente autorreportado de intento de suicidio alguna vez en la vida. "
        "No anticipa intentos futuros ni establece un diagnóstico. Utiliza únicamente perfiles ficticios."
    )
    info = load_mejor_modelo_info()
    if info.get("schema_version") != SCHEMA_VERSION:
        st.warning("Ejecuta el pipeline para generar un modelo compatible con el esquema de datos actual.")
        return
    model = load_mejor_modelo()
    if model is None:
        st.warning("No se encuentra el modelo de la ejecución activa.")
        return
    features = info["features"]
    unknown = set(features) - set(FEATURES)
    if unknown:
        st.error(f"El formulario no reconoce estos predictores: {sorted(unknown)}")
        return
    age, female, sexual_history, values = read_profile(features)
    if not st.button("Clasificar perfil", type="primary"):
        return
    try:
        frame = prediction_frame(features, values, age=age, female=female, sexual_history=sexual_history)
        if frame.isna().all(axis=1).any():
            st.warning("Introduce información antes de clasificar el perfil.")
            return
        show_prediction(frame, model)
    except (ValueError, KeyError) as error:
        st.error(str(error))
