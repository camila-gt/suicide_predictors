from pathlib import Path

import numpy as np
import pandas as pd

from suicide_predictors.schema import FEATURES, NOT_APPLICABLE, TARGET

VIOLENT_ITEMS = tuple(f"d0901{c}" for c in "cdfghijk")
CESD_ITEMS = tuple(f"d0601{c}" for c in "abcdefg")
FOOD_ITEMS = tuple(f"d06a{i}" for i in range(1, 11))
SUBSTANCE_BLOCKS = {
    "drogas_ilegales": tuple(f"d01a08{c}" for c in "abcdefghi"),
    "enfermedad_cronica": tuple(f"d02o1{c}" for c in "abcd"),
}
MEDICATION_USE_ITEMS = tuple(f"d01a03{c}" for c in "abcd")
MEDICATION_MODE_ITEMS = tuple(f"d01a04{c}" for c in "abcd")
FUNCTIONING_BLOCKS = {
    "deficit_cognitivo": ("d0415", "d0416", "d0417"),
    "desregulacion_conductual": ("d0418", "d0419"),
    "aislamiento_social": ("d0420",),
}
METHOD_NAMES = (
    "medicamentos",
    "narcoticos",
    "alcohol",
    "hidrocarburos",
    "fumigantes",
    "quimicos",
    "ahorcamiento",
    "arma_fuego",
    "quemadura",
    "cortantes",
    "arrojarse",
)


def numeric(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").astype(float)


def responses(frame: pd.DataFrame, columns: list[str] | tuple[str, ...], valid: tuple[int, ...]) -> pd.DataFrame:
    values = frame.loc[:, list(columns)].apply(numeric)
    return values.where(values.isin(valid))


def any_positive(block: pd.DataFrame, positive: tuple[int, ...] = (1,)) -> pd.Series:
    result = pd.Series(np.nan, index=block.index, dtype=float)
    complete_block = block.notna().all(axis=1)
    positive_response = block.isin(positive).any(axis=1)
    result.loc[complete_block] = 0.0
    result.loc[positive_response] = 1.0
    return result


def binary(series: pd.Series, yes: tuple[int, ...] = (1,), no: tuple[int, ...] = (2,)) -> pd.Series:
    values = numeric(series)
    codes = dict.fromkeys(yes, 1.0)
    codes.update(dict.fromkeys(no, 0.0))
    return values.map(codes)


def require_columns(frame: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas de origen: {sorted(missing)}")


def validate_raw(frame: pd.DataFrame) -> None:
    require_columns(frame, ("edad", "sexo", "d0810", "d0819", "ponde_f", "upm", "estrato"))
    domains = {"edad": tuple(range(10, 20)), "sexo": (1, 2), "d0810": (1, 2, 3, 8, 9), "d0819": (1, 2, 3)}
    for column, domain in domains.items():
        values = numeric(frame[column])
        invalid_code = values.notna() & ~values.isin(domain)
        original_text = frame[column].astype("string").str.strip().fillna("")
        malformed_number = original_text.ne("") & values.isna()
        invalid_rows = invalid_code | malformed_number
        if invalid_rows.any():
            raise ValueError(f"Códigos inesperados en {column}: {int(invalid_rows.sum())} registros")
    for column in ("edad", "sexo", "ponde_f", "estrato"):
        values = numeric(frame[column])
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{column} contiene valores ausentes o no finitos")
    missing_upm = frame["upm"].isna().any()
    empty_upm = frame["upm"].astype("string").str.strip().eq("").any()
    if missing_upm or empty_upm:
        raise ValueError("Identificadores de UPM ausentes")
    if numeric(frame["ponde_f"]).le(0).any():
        raise ValueError("Los factores de expansión deben ser positivos")
    if "FOLIO_I" in frame and frame["FOLIO_I"].duplicated().any():
        raise ValueError("Existen folios individuales duplicados")
    if "entidad" in frame:
        entities_per_upm = frame.groupby("upm")["entidad"].nunique()
        if entities_per_upm.gt(1).any():
            raise ValueError("Los identificadores de UPM no son únicos entre entidades")


def analytic_rows(raw: pd.DataFrame) -> pd.DataFrame:
    validate_raw(raw)
    valid_asi = numeric(raw["d0810"]).isin((1, 2, 3))
    valid_outcome = numeric(raw["d0819"]).isin((1, 2, 3))
    keep = valid_asi & valid_outcome
    if not keep.any():
        raise ValueError("No hay registros con ASI y desenlace válidos")
    return raw.loc[keep].copy()


def add_symptom_scores(data: pd.DataFrame, output: pd.DataFrame) -> None:
    cesd = responses(data, CESD_ITEMS, (1, 2, 3, 4)) - 1
    cesd["d0601f"] = 3 - cesd["d0601f"]
    output["depresion_cesd"] = cesd.sum(axis=1, min_count=7)
    food_responses = responses(data, FOOD_ITEMS, (1, 2, 3, 4))
    output["conducta_alim"] = food_responses.sum(axis=1, min_count=10)


def nonprescribed_medication_use(data: pd.DataFrame) -> pd.Series:
    """Combina consumo (1.a.3) y forma de uso (1.a.4) para cada grupo de medicamentos."""
    require_columns(data, (*MEDICATION_USE_ITEMS, *MEDICATION_MODE_ITEMS))
    usage = responses(data, MEDICATION_USE_ITEMS, (1, 2))
    modes = responses(data, MEDICATION_MODE_ITEMS, (1, 2))
    classified = pd.DataFrame(np.nan, index=data.index, columns=MEDICATION_USE_ITEMS)
    for use_column, mode_column in zip(MEDICATION_USE_ITEMS, MEDICATION_MODE_ITEMS, strict=True):
        used = usage[use_column].eq(1)
        not_used = usage[use_column].eq(2)
        used_as_prescribed = used & modes[mode_column].eq(2)
        used_without_or_against_prescription = used & modes[mode_column].eq(1)
        classified.loc[not_used | used_as_prescribed, use_column] = 2.0
        classified.loc[used_without_or_against_prescription, use_column] = 1.0
    return any_positive(classified)


def add_substance_indicators(data: pd.DataFrame, output: pd.DataFrame, female: pd.Series) -> None:
    output["fuma_actual"] = binary(data["d0101"], (1, 2), (3,))
    output["vapea_actual"] = binary(data["d0107"], (1, 2), (3, 4))
    output["alcohol_12m"] = binary(data["d0108"], (1, 2, 3, 4), (5, 6))
    male_binge = binary(data["d0111"])
    female_binge = binary(data["d0112"])
    output["binge_alcohol"] = male_binge.where(~female, female_binge)
    no_alcohol_consumption = output["alcohol_12m"].eq(0)
    output.loc[no_alcohol_consumption, "binge_alcohol"] = 0.0
    male_annual_binge = numeric(data["d0109"])
    female_annual_binge = numeric(data["d0110"])
    annual_binge = male_annual_binge.where(~female, female_annual_binge)
    no_annual_binge = annual_binge.eq(6)
    missing_monthly_binge = output["binge_alcohol"].isna()
    output.loc[no_annual_binge & missing_monthly_binge, "binge_alcohol"] = 0.0
    output["sustancias_medicas"] = nonprescribed_medication_use(data)
    for name, columns in SUBSTANCE_BLOCKS.items():
        block = responses(data, columns, (1, 2))
        output[name] = any_positive(block)


def child_walking_difficulty(data: pd.DataFrame) -> pd.Series:
    """Recupera la severidad a 100 m cuando el salto omite la pregunta de 500 m."""
    uses_assistance = numeric(data["d0403"]).eq(1)
    assisted_100m = numeric(data["d0406"])
    unassisted_100m = numeric(data["d0410"])
    walking_100m = assisted_100m.where(uses_assistance, unassisted_100m)
    assisted_500m = numeric(data["d0407"])
    unassisted_500m = numeric(data["d0411"])
    walking_500m = assisted_500m.where(uses_assistance, unassisted_500m)
    severe_at_100m = walking_100m.isin((3, 4))
    return walking_500m.mask(severe_at_100m, walking_100m)


def physical_disability(data: pd.DataFrame, age: pd.Series) -> pd.Series:
    child = age.lt(18)
    child_walking = child_walking_difficulty(data)
    adult_walking = numeric(data["d0425"])
    walking = child_walking.where(child, adult_walking)
    child_care = numeric(data["d0412"])
    adult_care = numeric(data["d0427"])
    care = child_care.where(child, adult_care)
    mobility = pd.DataFrame({"walking": walking, "care": care})
    mobility = mobility.where(mobility.isin((1, 2, 3, 4)))
    return any_positive(mobility, (3, 4))


def add_functioning_indicators(data: pd.DataFrame, output: pd.DataFrame) -> None:
    output["discapacidad_fisica"] = physical_disability(data, output["edad"])
    anxiety = responses(data, ("d0421",), (1, 2, 3, 4, 5))
    output["ansiedad_frec"] = 6 - anxiety["d0421"]
    for name, columns in FUNCTIONING_BLOCKS.items():
        block = responses(data, columns, (1, 2, 3, 4))
        output[name] = any_positive(block, (3, 4))


def mark_structural_absence(data: pd.DataFrame, output: pd.DataFrame, female: pd.Series) -> None:
    age = output["edad"]
    no_sexual_history = numeric(data["d0305"]).eq(0)
    for name, spec in FEATURES.items():
        applicable = age.between(spec.min_age, spec.max_age)
        if spec.female_only:
            applicable &= female
        if spec.sexual_history:
            applicable &= age.ge(12) & ~no_sexual_history
        output.loc[~applicable, name] = NOT_APPLICABLE


def build_ml(raw: pd.DataFrame) -> pd.DataFrame:
    data = analytic_rows(raw)
    require_columns(data, (*CESD_ITEMS, *FOOD_ITEMS, *VIOLENT_ITEMS, "d0305"))
    output = pd.DataFrame(index=data.index)
    output["edad"] = numeric(data["edad"])
    female = numeric(data["sexo"]).eq(2)
    output["asi"] = binary(data["d0810"], (1, 2), (3,))
    add_symptom_scores(data, output)
    add_substance_indicators(data, output, female)
    add_functioning_indicators(data, output)
    family_violence = responses(data, VIOLENT_ITEMS, (1, 2))
    output["violencia_familiar"] = any_positive(family_violence)
    output["relacion_forzada"] = binary(data["d0309"])
    output["embarazo"] = binary(data["d0313"])
    output["violencia_reciente"] = binary(data["d0801"])
    output["sexo_Mujer"] = female.astype(float)
    mark_structural_absence(data, output, female)
    output = output.loc[:, list(FEATURES)]
    output[TARGET] = binary(data["d0819"], (1, 2), (3,)).astype(int)
    output["row_id"] = data.index.astype(int)
    output["upm"] = data["upm"].astype("string").str.strip()
    for column in ("estrato", "ponde_f"):
        output[column] = numeric(data[column])
    return output.reset_index(drop=True)


def add_outcome_indicators(data: pd.DataFrame, output: pd.DataFrame) -> None:
    indicators = (
        ("asi", "d0810", (1, 2), (3,)),
        ("ideacion", "d0817", (1,), (2,)),
        ("intento", "d0819", (1, 2), (3,)),
    )
    for name, source, yes, no in indicators:
        output[name] = binary(data[source], yes, no)
        if name == "asi":
            labels = {0: "Sin ASI", 1: "Con ASI"}
        else:
            labels = {0: f"Sin {name}", 1: f"Con {name}"}
        output[f"{name}_label"] = output[name].map(labels)


def add_attempt_methods(data: pd.DataFrame, output: pd.DataFrame) -> None:
    method_columns = []
    for column in data:
        if column.startswith("D0821") and not column.endswith("E"):
            method_columns.append(column)
    block = data[method_columns].apply(numeric)
    block = block.where(block.isin(range(1, 12)))
    has_method_response = block.notna().any(axis=1)
    reported_attempt = output["intento"].eq(1)
    for code, name in enumerate(METHOD_NAMES, 1):
        reported_method = block.eq(code).any(axis=1)
        value = reported_method.astype(float).where(has_method_response)
        output[f"metodo_{name}"] = value.where(reported_attempt)


def build_descriptive(raw: pd.DataFrame) -> pd.DataFrame:
    data = analytic_rows(raw)
    output = pd.DataFrame(index=data.index)
    rename = {
        "d0810": "abuso_sexual",
        "d0811": "sexo_agresor",
        "d0812": "parentesco_agresor",
        "d0819": "intento_suicidio",
        "d0801": "violencia_reciente",
        "d0813": "atencion_tras_asi",
        "d0814": "denuncia_asi",
        "d0815": "razon_no_denuncia",
        "ponde_f": "ponderador",
    }
    for source, target in rename.items():
        output[target] = numeric(data[source])
    domains = {
        "sexo_agresor": (1, 2),
        "parentesco_agresor": tuple(range(1, 8)),
        "violencia_reciente": (1, 2),
        "atencion_tras_asi": tuple(range(1, 9)),
        "denuncia_asi": (1, 2),
        "razon_no_denuncia": (1, 2, 3, 4),
    }
    for column, valid in domains.items():
        output[column] = output[column].where(output[column].isin(valid))
    output["edad"] = numeric(data["edad"])
    output["sexo"] = numeric(data["sexo"]).map({1: "Hombre", 2: "Mujer"})
    add_outcome_indicators(data, output)
    output["grupo_edad"] = pd.cut(output["edad"], (9, 12, 15, 19), labels=("10-12", "13-15", "16-19"))
    add_attempt_methods(data, output)
    output["row_id"] = data.index.astype(int)
    output["upm"] = data["upm"].astype("string").str.strip()
    for column in ("estrato", "entidad"):
        output[column] = numeric(data[column])
    return output.reset_index(drop=True)


def prepare(raw_path: Path, workspace: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from suicide_predictors.plots import plot_target_distribution

    raw = pd.read_csv(raw_path, sep=";", encoding="utf-8-sig", low_memory=False)
    descriptive = build_descriptive(raw)
    ml = build_ml(raw)
    directory = workspace / "data"
    directory.mkdir(parents=True, exist_ok=True)
    descriptive.to_csv(directory / "adolescentes_limpio.csv", index=False, encoding="utf-8-sig")
    ml.to_csv(directory / "adolescentes_ml.csv", index=False, encoding="utf-8-sig")
    feature_values = ml[list(FEATURES)]
    quality = pd.DataFrame(
        {
            "feature": list(FEATURES),
            "missing": feature_values.isna().sum().values,
            "not_applicable": feature_values.eq(NOT_APPLICABLE).sum().values,
        }
    )
    quality.to_csv(directory / "calidad_predictores.csv", index=False)
    target_figure = workspace / "figuras" / "preprocesamiento" / "14_target_intento_frecuencias.png"
    plot_target_distribution(ml[TARGET], target_figure)
    return descriptive, ml
