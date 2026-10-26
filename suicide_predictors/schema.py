from dataclasses import dataclass

NOT_APPLICABLE = -1.0
TARGET = "intento_bin"
DESIGN_COLUMNS = ("row_id", "upm", "estrato", "ponde_f")
FEATURE_CHANNELS = ("value", "not_applicable", "unknown")


@dataclass(frozen=True)
class FeatureSpec:
    label: str
    minimum: int = 0
    maximum: int = 1
    default: int = 0
    max_age: int = 19
    sexual_history: bool = False
    female_only: bool = False
    min_age: int = 10


FEATURES = {
    "edad": FeatureSpec("Edad (años)", 10, 19, 14),
    "asi": FeatureSpec("Antecedente autorreportado de abuso sexual"),
    "depresion_cesd": FeatureSpec("Sintomatología depresiva (CES-D 7, última semana)", 0, 21, 5),
    "conducta_alim": FeatureSpec("Conductas alimentarias (últimos tres meses)", 10, 40, 15),
    "fuma_actual": FeatureSpec("Consumo actual de cigarrillos"),
    "vapea_actual": FeatureSpec("Consumo actual de vapeadores"),
    "alcohol_12m": FeatureSpec("Consumo de alcohol en los últimos 12 meses"),
    "binge_alcohol": FeatureSpec("Consumo de 5+ copas (hombres) o 4+ (mujeres) en una ocasión, últimos 30 días"),
    "drogas_ilegales": FeatureSpec("Uso de sustancias no médicas alguna vez", min_age=12),
    "sustancias_medicas": FeatureSpec(
        "Uso de medicamentos sin receta o distinto de lo indicado, alguna vez",
        min_age=12,
    ),
    "enfermedad_cronica": FeatureSpec("Diagnóstico de alguna enfermedad crónica del bloque estudiado"),
    "relacion_forzada": FeatureSpec("Primera relación sexual presionada o forzada", sexual_history=True),
    "embarazo": FeatureSpec("Embarazo alguna vez", sexual_history=True, female_only=True),
    "violencia_reciente": FeatureSpec("Daño a la salud por violencia en los últimos 12 meses"),
    "discapacidad_fisica": FeatureSpec("Dificultad severa para movilidad o cuidado propio"),
    "ansiedad_frec": FeatureSpec("Frecuencia de ansiedad", 1, 5, 2, max_age=17),
    "deficit_cognitivo": FeatureSpec("Dificultad severa para aprender, recordar o concentrarse", max_age=17),
    "desregulacion_conductual": FeatureSpec("Dificultad severa para controlar conducta o aceptar rutinas", max_age=17),
    "aislamiento_social": FeatureSpec("Dificultad severa para hacer amigos", max_age=17),
    "violencia_familiar": FeatureSpec("Disciplina física o psicológica violenta en el último mes", max_age=14),
    "sexo_Mujer": FeatureSpec("Sexo registrado en la encuesta: Mujer"),
}

ANXIETY_OPTIONS = {1: "Nunca", 2: "Varias veces al año", 3: "Mensualmente", 4: "Semanalmente", 5: "Diariamente"}


def is_applicable(name: str, age: int, female: bool, sexual_history: bool | None = None) -> bool:
    spec = FEATURES[name]
    if age < spec.min_age or age > spec.max_age:
        return False
    if spec.female_only and not female:
        return False
    if spec.sexual_history:
        if age < 12 or sexual_history is False:
            return False
    return True
