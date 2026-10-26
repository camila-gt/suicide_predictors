from collections.abc import Mapping

import numpy as np
import pandas as pd

from suicide_predictors.schema import FEATURES, NOT_APPLICABLE, is_applicable


def prediction_frame(
    features: list[str],
    values: Mapping[str, float],
    *,
    age: int,
    female: bool,
    sexual_history: bool | None = None,
) -> pd.DataFrame:
    unknown = set(features) - set(FEATURES)
    missing = set(features) - set(values)
    if unknown or missing:
        raise ValueError(
            f"Contrato de entrada incompatible: desconocidos={sorted(unknown)}, faltantes={sorted(missing)}"
        )
    if not 10 <= age <= 19:
        raise ValueError("La edad debe estar entre 10 y 19 años")
    row = {}
    for name in features:
        spec = FEATURES[name]
        value = float(values[name])
        if name == "asi" and np.isnan(value):
            raise ValueError("El antecedente de ASI no se imputa; selecciona una respuesta para ese predictor")
        applicable = is_applicable(name, age, female, sexual_history)
        if not applicable:
            if value != NOT_APPLICABLE:
                raise ValueError(f"{name} no aplica al perfil indicado")
        elif not np.isnan(value):
            finite = np.isfinite(value)
            within_range = spec.minimum <= value <= spec.maximum
            whole_number = value.is_integer()
            if not finite or not within_range or not whole_number:
                raise ValueError(f"{name} está fuera de su dominio")
        row[name] = value
    if "edad" in row and row["edad"] != age:
        raise ValueError("La edad del predictor no coincide con el perfil")
    if "sexo_Mujer" in row and row["sexo_Mujer"] != float(female):
        raise ValueError("El sexo del predictor no coincide con el perfil")
    return pd.DataFrame([row], columns=features)
