from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve


def save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    save_exploration(fig, path)


def bar(values: pd.Series, title: str, xlabel: str, path: Path) -> None:
    with plt.rc_context(MODEL_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        values.sort_values(kind="stable").plot.barh(ax=ax, color=PALETTE["blue"])
        ax.set(title=title, xlabel=xlabel, ylabel="")
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, path)


def weighted_categories(data: pd.DataFrame, column: str, labels: dict | None = None) -> pd.Series:
    valid = data.loc[data[column].notna()]
    values = valid.groupby(column, observed=True)["ponderador"].sum()
    if labels:
        values.index = values.index.map(labels)
    total_weight = values.sum()
    if total_weight == 0:
        return values
    return 100 * values / total_weight


def prevalence(data: pd.DataFrame, column: str) -> float:
    valid = data.loc[data[column].notna()]
    denominator = valid["ponderador"].sum()
    if denominator == 0:
        return np.nan
    positive_weight = valid.loc[valid[column].eq(1), "ponderador"].sum()
    return float(100 * positive_weight / denominator)


EXPLORATION_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
}
PALETTE = {
    "blue": "#89b4fa",
    "pink": "#f5c2e7",
    "green": "#a6e3a1",
    "peach": "#fab387",
    "mauve": "#cba6f7",
    "red": "#f38ba8",
    "lavender": "#b4befe",
    "sapphire": "#74c7ec",
}
SEX_COLORS = {"Hombre": PALETTE["blue"], "Mujer": PALETTE["pink"]}
GROUP_COLORS = {**SEX_COLORS, "Total": PALETTE["green"]}
MODEL_STYLE = {
    **EXPLORATION_STYLE,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
}
MODEL_COLORS = {
    "Regresión Logística": "#7c3aed",
    "Gradient Boosting": "#0ea5e9",
    "SVM (RBF)": "#f59e0b",
    "Random Forest": "#10b981",
}
SELECTION_COLORS = {"curve": "#2171B5", "band": "#6BAED6", "selected": "#C23B22"}
METHOD_LABELS = {
    "metodo_medicamentos": "Envenenamiento con medicamentos",
    "metodo_narcoticos": "Envenenamiento con narcóticos",
    "metodo_alcohol": "Envenenamiento con alcohol",
    "metodo_hidrocarburos": "Inhalación de hidrocarburos",
    "metodo_fumigantes": "Fumigantes o insecticidas",
    "metodo_quimicos": "Productos químicos, ácidos o corrosivos",
    "metodo_ahorcamiento": "Ahorcamiento",
    "metodo_arma_fuego": "Arma de fuego",
    "metodo_quemadura": "Quemaduras",
    "metodo_cortantes": "Objetos cortantes",
    "metodo_arrojarse": "Arrojarse al vacío o a un vehículo en movimiento",
}


def integer_label(value: float) -> str:
    return f"{int(value):,}".replace(",", " ")


def percentage_label(value: float) -> str:
    if 0 < value < 0.1:
        return "<0,1 %"
    return f"{value:.1f} %".replace(".", ",")


def percentage_tick(value: float, _position: int) -> str:
    return f"{value:g} %".replace(".", ",")


def exploration_panels(
    title: str, note: str, *, width: float = 7, height: float = 4.5, panels: int = 1
) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(1, panels, figsize=(width, height), squeeze=False, layout="constrained")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    wrapped_note = fill(note, width=int(width * 14))
    fig.supxlabel(f"{wrapped_note}\nFuente: ENSANUT Continua 2024.", fontsize=8.5, fontstyle="italic")
    return fig, axes[0]


def exploration_axes(title: str, note: str, height: float) -> tuple[Figure, Axes]:
    fig, axes = exploration_panels(title, note, height=height)
    return fig, axes[0]


def save_exploration(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=300, facecolor="white")
    finally:
        plt.close(fig)


def exploration_bars(
    values: pd.Series,
    title: str,
    path: Path,
    *,
    note: str,
    percentage: bool = True,
    sort: bool = True,
    colors: dict[str, str] | None = None,
) -> None:
    values = values.dropna()
    if sort:
        values = values.sort_values(ascending=False, kind="stable")
    height = max(3.6, 1.9 + 0.48 * len(values))
    with plt.rc_context(EXPLORATION_STYLE):
        fig, ax = exploration_axes(title, note, height)
        if values.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Sin respuestas disponibles para esta figura.", ha="center", transform=ax.transAxes)
        else:
            positions = np.arange(len(values))
            labels = [fill(str(label), width=29) for label in values.index]
            bar_colors = []
            for label in values.index:
                color = PALETTE["blue"]
                if colors is not None:
                    color = colors.get(str(label), color)
                bar_colors.append(color)
            bars = ax.barh(positions, values.to_numpy(), height=0.58, color=bar_colors)
            ax.set_yticks(positions, labels)
            ax.invert_yaxis()
            ax.margins(y=0.08)
            ax.set_xlim(0, max(float(values.max()) * 1.22, 1.0))
            ax.grid(False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=not percentage))
            if percentage:
                annotations = [percentage_label(value) for value in values]
                ax.xaxis.set_major_formatter(FuncFormatter(percentage_tick))
                ax.set_xlabel("Porcentaje ponderado")
            else:
                annotations = [integer_label(value) for value in values]
                ax.set_xlabel("Frecuencia (n, sin ponderar)")
            ax.bar_label(bars, labels=annotations, padding=7, fontsize=10)
        save_exploration(fig, path)


def plot_age_distribution(data: pd.DataFrame, path: Path) -> None:
    ages = np.arange(10, 20)
    counts = data.groupby(["edad", "sexo"], observed=True).size().unstack("sexo", fill_value=0)
    counts = counts.reindex(index=ages, columns=list(SEX_COLORS), fill_value=0)
    note = f"Frecuencias sin ponderar. Muestra analítica: n = {integer_label(len(data))}."
    with plt.rc_context(EXPLORATION_STYLE):
        fig, axes = exploration_panels("Distribución de edad por sexo", note, width=10, height=4.5)
        ax = axes[0]
        for position, (sex, color) in enumerate(SEX_COLORS.items()):
            offset = (position - 0.5) * 0.4
            ax.bar(
                ages + offset,
                counts[sex],
                width=0.37,
                label=sex,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
            )
        ax.set_xticks(ages)
        ax.set(xlabel="Edad (años)", ylabel="Participantes (n, sin ponderar)")
        maximum_count = float(counts.to_numpy().max())
        ax.set_ylim(0, max(maximum_count * 1.25, 1.0))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.legend(loc="upper right")
        save_exploration(fig, path)


def draw_pie(ax: Axes, values: pd.Series, colors: list[str]) -> None:
    values = values.dropna()
    if values.empty or values.sum() <= 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Sin respuestas disponibles", ha="center", transform=ax.transAxes)
        return
    ax.pie(
        values.to_numpy(),
        labels=values.index,
        colors=colors,
        autopct=percentage_label,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )


def plot_sex_distribution(data: pd.DataFrame, path: Path) -> None:
    values = weighted_categories(data, "sexo")
    colors = [SEX_COLORS[label] for label in values.index]
    values = values.rename(index={"Hombre": "Hombres", "Mujer": "Mujeres"})
    title = f"Distribución por sexo\n(n = {integer_label(len(data))})"
    note = "Porcentajes ponderados con el factor de expansión. n: tamaño de la muestra sin ponderar."
    with plt.rc_context(EXPLORATION_STYLE):
        fig, axes = exploration_panels(title, note, width=5.5, height=5.5)
        draw_pie(axes[0], values, colors)
        save_exploration(fig, path)


def prevalence_groups(data: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for label in GROUP_COLORS:
        group = data
        if label != "Total":
            group = data.loc[data["sexo"].eq(label)]
        rows.append(
            {
                "group": label,
                "percentage": prevalence(group, column),
                "positive": int(group[column].eq(1).sum()),
                "denominator": int(group[column].notna().sum()),
            }
        )
    return pd.DataFrame(rows).set_index("group")


def prevalence_note(summary: pd.DataFrame) -> str:
    bases = []
    for label, row in summary.iterrows():
        bases.append(f"{label}: n = {integer_label(row['denominator'])}")
    return "Porcentajes ponderados; n en barras: casos positivos sin ponderar. Bases válidas: " + "; ".join(bases) + "."


def draw_prevalence(ax: Axes, summary: pd.DataFrame) -> None:
    valid = summary.dropna(subset=["percentage"])
    bars = ax.bar(
        valid.index,
        valid["percentage"],
        color=[GROUP_COLORS[label] for label in valid.index],
        edgecolor="white",
    )
    annotations = []
    for row in valid.itertuples():
        annotations.append(f"{percentage_label(row.percentage)}\n(n={integer_label(row.positive)})")
    ax.bar_label(bars, labels=annotations, padding=5, fontsize=10)
    maximum = valid["percentage"].max()
    if pd.isna(maximum):
        maximum = 0.0
    ax.set_ylim(0, max(float(maximum) * 1.45, 1.0))
    ax.set_ylabel("Porcentaje ponderado")
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_tick))


def plot_attempt_frequency(data: pd.DataFrame, path: Path) -> None:
    summary = prevalence_groups(data, "intento")
    note = prevalence_note(summary) + " Panel derecho: frecuencias sin ponderar."
    with plt.rc_context(EXPLORATION_STYLE):
        fig, axes = exploration_panels("", note, width=12, height=4.8, panels=2)
        draw_prevalence(axes[0], summary)
        axes[0].set_title("Intentos de suicidio por sexo\n(porcentajes ponderados)")
        positions = np.arange(len(SEX_COLORS))
        for offset, code, label, color in (
            (-0.175, 1, "1 vez", PALETTE["blue"]),
            (0.175, 2, "2 o más veces", PALETTE["peach"]),
        ):
            counts = []
            for sex in SEX_COLORS:
                group = data.loc[data["sexo"].eq(sex)]
                counts.append(int(group["intento_suicidio"].eq(code).sum()))
            axes[1].bar(positions + offset, counts, width=0.35, label=label, color=color, edgecolor="white")
        axes[1].set_xticks(positions, list(SEX_COLORS))
        axes[1].set(title="Frecuencia de intentos por sexo", ylabel="Frecuencia (n, sin ponderar)")
        axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].legend()
        save_exploration(fig, path)


def plot_prevalence_by_sex(data: pd.DataFrame, directory: Path) -> None:
    plots = (
        (3, "asi", "Prevalencia de ASI por sexo", "prevalencia_asi_sexo"),
        (7, "ideacion", "Ideación suicida por sexo", "ideacion_suicida"),
        (11, "violencia_reciente", "Violencia en los últimos 12 meses por sexo", "violencia_general"),
    )
    for number, column, title, filename in plots:
        summary = prevalence_groups(data, column)
        with plt.rc_context(EXPLORATION_STYLE):
            fig, ax = exploration_axes(title + "\n(porcentajes ponderados)", prevalence_note(summary), 4.8)
            draw_prevalence(ax, summary)
            save_exploration(fig, directory / f"{number:02d}_{filename}.png")
    plot_attempt_frequency(data, directory / "08_intentos_suicidio_sexo.png")


def draw_horizontal_counts(ax: Axes, counts: pd.Series, color: str) -> None:
    positions = np.arange(len(counts))
    bars = ax.barh(positions, counts, color=color, edgecolor="white")
    ax.set_yticks(positions, [fill(str(label), width=27) for label in counts.index])
    ax.set_xlim(0, max(float(counts.max()) * 1.2, 1.0))
    ax.set_xlabel("Frecuencia (n, sin ponderar)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.bar_label(bars, labels=[integer_label(value) for value in counts], padding=3, fontsize=9)


def count_comparison_panels(
    totals: pd.Series, by_sex: pd.DataFrame, path: Path, *, title: str, note: str, color: str
) -> None:
    totals = totals.sort_values(kind="stable")
    by_sex = by_sex.reindex(index=totals.index, columns=list(SEX_COLORS), fill_value=0)
    height = max(5.0, 1.9 + 0.48 * len(totals))
    with plt.rc_context(EXPLORATION_STYLE):
        fig, axes = exploration_panels("", note, width=13, height=height, panels=2)
        draw_horizontal_counts(axes[0], totals, color)
        axes[0].set_title(title + "\n(Total)")
        positions = np.arange(len(totals))
        for position, (sex, sex_color) in enumerate(SEX_COLORS.items()):
            offset = (position - 0.5) * 0.36
            axes[1].barh(positions + offset, by_sex[sex], height=0.36, label=sex, color=sex_color, edgecolor="white")
        axes[1].set_yticks(positions, [fill(str(label), width=27) for label in totals.index])
        axes[1].set(title=title + " por sexo", xlabel="Frecuencia (n, sin ponderar)")
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].legend(title="Sexo del adolescente")
        save_exploration(fig, path)


def plot_asi_age(victims: pd.DataFrame, path: Path) -> None:
    labels = {1: "Antes de 12", 2: "12 o más"}
    note = "ASI: abuso sexual autorreportado. Porcentajes ponderados entre respuestas disponibles; n sin ponderar."
    with plt.rc_context(EXPLORATION_STYLE):
        fig, axes = exploration_panels("Edad referida en el reporte de ASI", note, width=11, height=4.6, panels=3)
        for ax, label in zip(axes, GROUP_COLORS, strict=True):
            group = victims
            if label != "Total":
                group = victims.loc[victims["sexo"].eq(label)]
            values = weighted_categories(group, "abuso_sexual", labels)
            age_colors = {"Antes de 12": PALETTE["red"], "12 o más": PALETTE["blue"]}
            colors = [age_colors[age_label] for age_label in values.index]
            draw_pie(ax, values, colors)
            ax.set_title(f"{label}\n(n={integer_label(group['abuso_sexual'].notna().sum())})")
        save_exploration(fig, path)


def plot_denunciation(victims: pd.DataFrame, path: Path) -> None:
    counts = victims["denuncia_asi"].value_counts().reindex([1, 2], fill_value=0)
    reasons = {1: "Miedo", 2: "Vergüenza", 3: "Amenazas", 4: "No sabía que podía denunciar"}
    not_reported = victims.loc[victims["denuncia_asi"].eq(2)]
    reason_counts = not_reported["razon_no_denuncia"].map(reasons).dropna().value_counts().sort_values()
    note = "Frecuencias sin ponderar entre adolescentes con ASI. Motivos: respuestas disponibles en las categorías analizadas."
    with plt.rc_context(EXPLORATION_STYLE):
        fig, axes = exploration_panels("", note, width=12, height=4.8, panels=2)
        bars = axes[0].bar(
            ["Sí denunció", "No denunció"], counts, color=[PALETTE["green"], PALETTE["red"]], edgecolor="white"
        )
        axes[0].bar_label(bars, labels=[integer_label(value) for value in counts], padding=4)
        axes[0].set(title="¿Denunciaron el ASI?", ylabel="Frecuencia (n, sin ponderar)")
        axes[0].set_ylim(0, max(float(counts.max()) * 1.2, 1.0))
        axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        if not reason_counts.empty:
            draw_horizontal_counts(axes[1], reason_counts, PALETTE["mauve"])
        else:
            axes[1].text(0.5, 0.5, "Sin respuestas disponibles", ha="center", transform=axes[1].transAxes)
        axes[1].set_title("Razón por la que no denunció")
        save_exploration(fig, path)


def plot_victim_categories(victims: pd.DataFrame, directory: Path) -> None:
    plot_asi_age(victims, directory / "04_asi_edad_inicio.png")
    relation_labels = {
        1: "Pareja",
        2: "Familiar",
        3: "Amigo/a",
        4: "Novio/a",
        5: "Vecino/conocido",
        6: "Desconocido/a",
        7: "Policía",
    }
    relations = victims["parentesco_agresor"].map(relation_labels)
    totals = relations.value_counts()
    if not totals.empty:
        by_sex = pd.crosstab(relations, victims["sexo"])
        note = f"Frecuencias sin ponderar entre adolescentes con ASI; parentesco disponible: n = {integer_label(totals.sum())}."
        count_comparison_panels(
            totals,
            by_sex,
            directory / "05_asi_agresor_relacion.png",
            title="Relación con la persona agresora",
            note=note,
            color=PALETTE["lavender"],
        )
    values = weighted_categories(victims, "sexo_agresor", {1: "Hombre", 2: "Mujer"})
    n_valid = int(victims["sexo_agresor"].notna().sum())
    note = f"Porcentajes ponderados entre adolescentes con ASI y respuesta disponible: n = {integer_label(n_valid)} (sin ponderar)."
    with plt.rc_context(EXPLORATION_STYLE):
        fig, axes = exploration_panels("Sexo del agresor en ASI\n(porcentajes ponderados)", note, width=5.5, height=5.5)
        draw_pie(axes[0], values, [SEX_COLORS[label] for label in values.index])
        save_exploration(fig, directory / "06_asi_sexo_agresor.png")
    plot_denunciation(victims, directory / "12_denuncia_asi.png")
    attention_labels = {
        1: "Nadie lo atendió",
        2: "Remedios/automedicación",
        3: "Curandero/hierbero",
        4: "Huesero/sobador",
        5: "Encargado de la comunidad",
        6: "Psicólogo/terapeuta",
        7: "Médico/consultorio",
        8: "Clínica/hospital",
    }
    counts = victims["atencion_tras_asi"].map(attention_labels).value_counts()
    if not counts.empty:
        note = f"Frecuencias sin ponderar entre adolescentes con ASI; respuestas en las categorías analizadas: n = {integer_label(counts.sum())}."
        colors = dict.fromkeys(counts.index, PALETTE["sapphire"])
        exploration_bars(
            counts,
            "Atención recibida después del ASI",
            directory / "13_atencion_tras_asi.png",
            note=note,
            percentage=False,
            colors=colors,
        )


def plot_attempt_methods(data: pd.DataFrame, path: Path) -> None:
    method_columns = [column for column in data if column.startswith("metodo_")]
    attempts = data.loc[data["intento"].eq(1)]
    methods = attempts[method_columns].sum()
    if not methods.gt(0).any():
        return
    methods = methods.loc[methods.gt(0)].rename(index=METHOD_LABELS)
    by_sex = pd.DataFrame(index=methods.index)
    for sex in SEX_COLORS:
        group = attempts.loc[attempts["sexo"].eq(sex), method_columns]
        by_sex[sex] = group.sum().rename(index=METHOD_LABELS)
    n_valid = int(attempts[method_columns].notna().any(axis=1).sum())
    note = f"Frecuencias sin ponderar. Con antecedente de intento y método registrado: n = {integer_label(n_valid)}. "
    note += "Respuesta múltiple; las categorías no son excluyentes. Se muestran métodos con al menos un reporte."
    count_comparison_panels(
        methods, by_sex, path, title="Métodos de intento de suicidio", note=note, color=PALETTE["peach"]
    )


def plot_attempt_by_asi(data: pd.DataFrame, path: Path) -> None:
    values = []
    annotations = []
    for code in (1, 0):
        group = data.loc[data["asi"].eq(code)]
        estimate = prevalence(group, "intento")
        values.append(estimate)
        denominator = int(group["intento"].notna().sum())
        annotation = "Sin respuesta"
        if pd.notna(estimate):
            annotation = f"{percentage_label(estimate)}\n(n={integer_label(denominator)})"
        annotations.append(annotation)
    note = (
        "ASI: abuso sexual autorreportado. Porcentajes ponderados dentro de cada grupo; n: denominadores sin ponderar."
    )
    with plt.rc_context(EXPLORATION_STYLE):
        fig, ax = exploration_axes("Intento de suicidio según antecedente de ASI\n(porcentajes ponderados)", note, 4.8)
        bars = ax.bar(["Con ASI", "Sin ASI"], values, color=[PALETTE["red"], PALETTE["blue"]], edgecolor="white")
        ax.bar_label(bars, labels=annotations, padding=5, fontsize=10)
        finite_values = pd.Series(values).dropna()
        maximum = 0.0
        if not finite_values.empty:
            maximum = float(finite_values.max())
        ax.set_ylim(0, max(maximum * 1.45, 1.0))
        ax.set_ylabel("Porcentaje con intento de suicidio (ponderado)")
        ax.yaxis.set_major_formatter(FuncFormatter(percentage_tick))
        save_exploration(fig, path)


def explore(data: pd.DataFrame, workspace: Path) -> None:
    directory = workspace / "exploracion"
    rows = []
    for column, label in (
        ("asi", "ASI"),
        ("ideacion", "Ideación suicida"),
        ("intento", "Intento de suicidio"),
        ("violencia_reciente", "Violencia reciente"),
    ):
        rows.append(
            {
                "Variable": label,
                "Prevalencia_ponderada (%)": prevalence(data, column),
                "n_positivos": int(data[column].eq(1).sum()),
                "Denominador (n)": int(data[column].notna().sum()),
            }
        )
    output = workspace / "exploracion"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "prevalencias_crudas.csv", index=False)
    plot_sex_distribution(data, directory / "01_distribucion_sexo.png")
    plot_age_distribution(data, directory / "02_distribucion_edad.png")
    plot_prevalence_by_sex(data, directory)
    victims = data.loc[data["asi"].eq(1)]
    plot_victim_categories(victims, directory)
    plot_attempt_methods(data, directory / "09_metodos_suicidio.png")
    plot_attempt_by_asi(data, directory / "10_asi_vs_intento.png")


def plot_target_distribution(target: pd.Series, path: Path) -> None:
    counts = target.value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    with plt.rc_context(MODEL_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(["No intentó", "Intentó"], counts, color=[PALETTE["blue"], PALETTE["red"]], width=0.5)
        annotations = []
        for count in counts:
            label = integer_label(count)
            if total > 0:
                label += f"\n({percentage_label(100 * count / total)})"
            annotations.append(label)
        ax.bar_label(bars, labels=annotations, padding=3, fontsize=11)
        ax.set(
            ylabel="Frecuencia (n adolescentes, sin ponderar)",
            title="Distribución del intento de suicidio (target binario)\nENSANUT 2024 — adolescentes",
        )
        ax.set_ylim(0, max(float(counts.max()) * 1.2, 1.0))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, path)


def plot_confusion_matrices(comparison: pd.DataFrame, predictions: pd.DataFrame, directory: Path) -> None:
    with plt.rc_context(MODEL_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        scores = comparison.set_index("modelo")["MCC_media"]
        for ax, name in zip(axes.flat, comparison["modelo"], strict=True):
            ConfusionMatrixDisplay.from_predictions(
                predictions["y"],
                predictions[f"{name}_pred"],
                display_labels=["Nunca", "Intentó"],
                ax=ax,
                cmap="Blues",
                colorbar=False,
                values_format="d",
            )
            ax.set(title=f"{name}\nMCC CV = {scores[name]:.3f}", xlabel="Predicho", ylabel="Real")
        fig.suptitle("Matrices de confusión (predicciones fuera de pliegue sobre desarrollo)", fontsize=14)
        save(fig, directory / "16_matrices_confusion.png")


def plot_model_comparison(comparison: pd.DataFrame, directory: Path) -> None:
    ordered = comparison.sort_values("MCC_media", kind="stable")
    with plt.rc_context(MODEL_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(ordered["modelo"], ordered["MCC_media"], xerr=ordered["MCC_sd"], color=PALETTE["mauve"], capsize=4)
        ax.set(xlabel="MCC medio (± desviación entre pliegues)", title="Comparación de algoritmos — MCC")
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, directory / "17_comparacion_mcc.png")


def plot_roc_curves(comparison: pd.DataFrame, predictions: pd.DataFrame, directory: Path) -> None:
    with plt.rc_context(MODEL_STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))
        for name in comparison["modelo"]:
            probability = predictions[f"{name}_proba"]
            fpr, tpr, _ = roc_curve(predictions["y"], probability)
            auc = roc_auc_score(predictions["y"], probability)
            ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2, label=f"{name} (AUC agrupado = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=1.2, label="Clasificador aleatorio")
        ax.set(
            xlabel="Tasa de falsos positivos (1 − especificidad)",
            ylabel="Tasa de verdaderos positivos (sensibilidad)",
            title="Curvas ROC — comparación de algoritmos\n(predicciones fuera de pliegue sobre desarrollo)",
            xlim=(-0.01, 1.01),
            ylim=(-0.01, 1.01),
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, directory / "18_curvas_roc.png")


def plot_selection_curve(curve: pd.DataFrame, n_selected: int, path: Path) -> None:
    feature_counts = curve["n_features"]
    mean_score = curve["mean_test_score"]
    standard_deviation = curve["std_test_score"]
    with plt.rc_context(MODEL_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(feature_counts, mean_score, marker="o", color=SELECTION_COLORS["curve"])
        ax.fill_between(
            feature_counts,
            mean_score - standard_deviation,
            mean_score + standard_deviation,
            alpha=0.2,
            color=SELECTION_COLORS["band"],
        )
        ax.axvline(
            n_selected, ls="--", color=SELECTION_COLORS["selected"], label=f"Seleccionados: {n_selected} predictores"
        )
        ax.set(
            xlabel="Número de predictores",
            ylabel="MCC no ponderado (validación cruzada)",
            title="RFECV — selección de predictores (clasificación binaria)\nRandom Forest balanceado; CV por UPM sobre desarrollo",
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, path)


def shap_figures(importance: pd.Series, explanation, directory: Path, *, model_name: str) -> None:
    import shap

    bar(
        importance.sort_values(ascending=False, kind="stable").head(15),
        f"Importancia global de variables (SHAP)\nModelo: {model_name}",
        "Media |SHAP| (probabilidad de la clase positiva)",
        directory / "18_shap_importancia_global.png",
    )
    with plt.rc_context(MODEL_STYLE):
        ax = shap.plots.beeswarm(explanation, show=False, max_display=12, color=shap.plots.colors.red_blue)
        ax.set(
            title="Impacto de variables en «Intentó» (SHAP — clase positiva)",
            xlabel="Valor SHAP (impacto en la probabilidad de «Intentó»)",
        )
        save(ax.figure, directory / "19_shap_beeswarm.png")


def classification_figures(
    comparison: pd.DataFrame, predictions: pd.DataFrame, curve: pd.DataFrame, n_selected: int, workspace: Path
) -> None:
    directory = workspace / "clasificacion"
    plot_confusion_matrices(comparison, predictions, directory)
    plot_model_comparison(comparison, directory)
    plot_roc_curves(comparison, predictions, directory)
    selection_path = workspace / "seleccion" / "15_rfecv_seleccion.png"
    plot_selection_curve(curve, n_selected, selection_path)
