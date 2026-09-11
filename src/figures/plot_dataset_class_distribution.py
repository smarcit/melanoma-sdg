#!/usr/bin/env python3
"""Plot stacked dataset/class distribution from master metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"dataset", "superclass"}
DEFAULT_METADATA = Path("data/other/no_fitz/master_metadata.csv")
DEFAULT_OUTPUT = Path("reports/figures/fig_clases_por_dataset.png")
DEFAULT_CSV_OUTPUT = Path("reports/figures/dataset_class_distribution.csv")
SOURCE_NOTE = "Fuente: elaboración propia."

COLORS = {
    "dark_blue": "#1F4E79",
    "light_blue": "#D9EAF7",
    "gray": "#6C757D",
    "soft_green": "#E8F3EC",
    "soft_orange": "#F6E7D8",
}

BASE_PALETTE = [
    COLORS["dark_blue"],
    COLORS["light_blue"],
    COLORS["soft_green"],
    COLORS["soft_orange"],
    COLORS["gray"],
    "#7A9CC6",
    "#A7C4A0",
    "#D6A77A",
    "#495057",
    "#B8D8E8",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lee master_metadata.csv y genera un grafico de barras apiladas "
            "por dataset y superclass."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Ruta al CSV de metadatos. Por defecto: {DEFAULT_METADATA}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ruta del PNG de salida. Por defecto: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help=f"Ruta del CSV completo sin agrupar. Por defecto: {DEFAULT_CSV_OUTPUT}",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Representa porcentajes por dataset en lugar de conteos absolutos.",
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=9,
        help="Maximo de clases visibles antes de agrupar minoritarias como Other. Por defecto: 9.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_args(args: argparse.Namespace) -> None:
    if not args.metadata.exists():
        fail(f"No se encontro el archivo de metadatos: {args.metadata}")
    if not args.metadata.is_file():
        fail(f"La ruta de metadatos no es un archivo: {args.metadata}")
    if args.output.suffix.lower() != ".png":
        fail("--output debe terminar en .png para cumplir la exportacion obligatoria.")
    if args.max_classes < 2:
        fail("--max-classes debe ser mayor o igual que 2.")


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(metadata_path, low_memory=False)
    except Exception as exc:
        fail(f"No se pudo leer el CSV '{metadata_path}': {exc}")

    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        fail(
            "Faltan columnas obligatorias en el CSV: "
            + ", ".join(missing)
            + ". Columnas requeridas: dataset, superclass."
        )

    df = df.loc[:, ["dataset", "superclass"]].copy()
    df["dataset"] = df["dataset"].astype("string").str.strip()
    df["superclass"] = df["superclass"].astype("string").str.strip()
    df = df.dropna(subset=["dataset", "superclass"])
    df = df[(df["dataset"] != "") & (df["superclass"] != "")]

    if df.empty:
        fail("No hay filas validas con dataset y superclass para construir el grafico.")

    return df


def build_full_distribution(df: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        df.groupby(["dataset", "superclass"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values(["dataset", "count", "superclass"], ascending=[True, False, True])
    )

    totals = distribution.groupby("dataset", observed=True)["count"].transform("sum")
    distribution["percentage"] = distribution["count"] / totals * 100
    return distribution


def prepare_plot_table(distribution: pd.DataFrame, max_classes: int) -> tuple[pd.DataFrame, bool]:
    totals_by_class = distribution.groupby("superclass", observed=True)["count"].sum().sort_values(ascending=False)
    class_order = totals_by_class.index.tolist()
    grouped = len(class_order) > max_classes

    if grouped:
        visible_classes = class_order[: max_classes - 1]
        plot_df = distribution.copy()
        plot_df["plot_superclass"] = np.where(
            plot_df["superclass"].isin(visible_classes),
            plot_df["superclass"],
            "Other",
        )
    else:
        visible_classes = class_order
        plot_df = distribution.copy()
        plot_df["plot_superclass"] = plot_df["superclass"]

    plot_distribution = (
        plot_df.groupby(["dataset", "plot_superclass"], observed=True)["count"]
        .sum()
        .reset_index()
    )
    plot_distribution["plot_superclass"] = pd.Categorical(
        plot_distribution["plot_superclass"],
        categories=visible_classes + (["Other"] if grouped else []),
        ordered=True,
    )
    return plot_distribution, grouped


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": COLORS["gray"],
            "axes.linewidth": 0.8,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
        }
    )


def make_plot(plot_distribution: pd.DataFrame, normalized: bool, grouped: bool) -> plt.Figure:
    configure_style()

    table = plot_distribution.pivot_table(
        index="dataset",
        columns="plot_superclass",
        values="count",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]

    if normalized:
        values = table.div(table.sum(axis=1), axis=0) * 100
        ylabel = "Porcentaje (%)"
        subtitle_metric = "porcentajes por dataset"
    else:
        values = table
        ylabel = "Numero de imagenes"
        subtitle_metric = "conteos absolutos"

    fig_width = max(9.0, 0.65 * len(values.index) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))

    bottoms = np.zeros(len(values.index))
    palette = BASE_PALETTE

    for idx, column in enumerate(values.columns):
        heights = values[column].to_numpy(dtype=float)
        ax.bar(
            values.index,
            heights,
            bottom=bottoms,
            label=str(column),
            color=palette[idx % len(palette)],
            edgecolor="white",
            linewidth=0.6,
        )
        bottoms += heights

    title = "Distribucion de clases por dataset"
    subtitle = f"Barras apiladas en {subtitle_metric}"
    if grouped:
        subtitle += "; clases minoritarias agrupadas en Other solo para la figura"

    ax.set_title(title, fontweight="bold", color=COLORS["dark_blue"], pad=22)
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=COLORS["gray"],
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    if normalized:
        ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    ax.legend(
        title="Superclass",
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    fig.text(0.01, 0.012, SOURCE_NOTE, ha="left", va="bottom", fontsize=8, color=COLORS["gray"])
    fig.tight_layout(rect=(0, 0.035, 0.84, 0.94))
    return fig


def save_outputs(fig: plt.Figure, distribution: pd.DataFrame, output: Path, csv_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    distribution.to_csv(csv_output, index=False)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    svg_output = output.with_suffix(".svg")
    fig.savefig(svg_output, bbox_inches="tight")

    print(f"CSV completo guardado en: {csv_output}")
    print(f"Figura PNG guardada en: {output}")
    print(f"Figura SVG guardada en: {svg_output}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    df = load_metadata(args.metadata)
    distribution = build_full_distribution(df)
    plot_distribution, grouped = prepare_plot_table(distribution, args.max_classes)
    fig = make_plot(plot_distribution, args.normalized, grouped)
    save_outputs(fig, distribution, args.output, args.csv_output)
    plt.close(fig)


if __name__ == "__main__":
    main()
