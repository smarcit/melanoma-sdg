#!/usr/bin/env python3
"""Create a dataset-by-example image grid from master metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"dataset", "path", "superclass"}
DEFAULT_METADATA = Path("data/other/no_fitz/master_metadata.csv")
DEFAULT_OUTPUT = Path("reports/figures/fig_grid_datasets.png")
SOURCE_NOTE = "Fuente: elaboración propia."

COLORS = {
    "dark_blue": "#1F4E79",
    "light_blue": "#D9EAF7",
    "gray": "#6C757D",
    "soft_green": "#E8F3EC",
    "soft_orange": "#F6E7D8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lee master_metadata.csv y genera una figura compuesta con ejemplos "
            "reales de cada dataset."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Ruta al CSV de metadatos. Por defecto: {DEFAULT_METADATA}",
    )
    parser.add_argument(
        "--n-per-dataset",
        type=int,
        default=5,
        help="Numero de imagenes por dataset. Por defecto: 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo aleatorio reproducible. Por defecto: 42.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ruta del PNG de salida. Por defecto: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_inputs(metadata_path: Path, n_per_dataset: int) -> None:
    if n_per_dataset < 1:
        fail("--n-per-dataset debe ser un entero mayor o igual que 1.")
    if not metadata_path.exists():
        fail(f"No se encontro el archivo de metadatos: {metadata_path}")
    if not metadata_path.is_file():
        fail(f"La ruta de metadatos no es un archivo: {metadata_path}")


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
            + ". Columnas requeridas: dataset, path, superclass."
        )

    df = df.copy()
    df["dataset"] = df["dataset"].astype("string").str.strip()
    df["path"] = df["path"].astype("string").str.strip()
    df["superclass"] = df["superclass"].astype("string").str.strip()
    df = df.dropna(subset=["dataset", "path", "superclass"])
    df = df[(df["dataset"] != "") & (df["path"] != "") & (df["superclass"] != "")]

    if "is_corrupt" in df.columns:
        corrupt = df["is_corrupt"].astype("string").str.lower().isin({"true", "1", "yes", "si"})
        df = df.loc[~corrupt].copy()

    if df.empty:
        fail("No hay filas validas tras filtrar dataset, path, superclass e is_corrupt.")

    return df


def resolve_image_path(path_value: str, metadata_path: Path) -> Path:
    image_path = Path(path_value)
    if image_path.is_absolute():
        return image_path

    repo_relative = Path.cwd() / image_path
    if repo_relative.exists():
        return repo_relative

    metadata_relative = metadata_path.parent / image_path
    return metadata_relative


def read_image(image_path: Path) -> np.ndarray | None:
    if not image_path.exists() or not image_path.is_file():
        return None
    try:
        image = mpimg.imread(image_path)
    except Exception:
        return None
    if image.ndim not in (2, 3) or image.size == 0:
        return None
    return image


def sample_valid_examples(
    df: pd.DataFrame,
    metadata_path: Path,
    n_per_dataset: int,
    seed: int,
) -> list[tuple[str, list[tuple[np.ndarray, str]]]]:
    rng = np.random.default_rng(seed)
    examples: list[tuple[str, list[tuple[np.ndarray, str]]]] = []

    for dataset in sorted(df["dataset"].dropna().unique()):
        group = df.loc[df["dataset"] == dataset].copy()
        order = rng.permutation(len(group))
        selected: list[tuple[np.ndarray, str]] = []

        for _, row in group.iloc[order].iterrows():
            image_path = resolve_image_path(str(row["path"]), metadata_path)
            image = read_image(image_path)
            if image is None:
                continue
            selected.append((image, str(row["superclass"])))
            if len(selected) == n_per_dataset:
                break

        if selected:
            examples.append((str(dataset), selected))
        else:
            print(
                f"AVISO: se omite el dataset '{dataset}' porque no se encontraron "
                "imagenes validas.",
                file=sys.stderr,
            )

    if not examples:
        fail("No se encontro ninguna imagen valida para construir la figura.")

    return examples


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 8,
            "text.color": "#222222",
            "axes.edgecolor": COLORS["gray"],
        }
    )


def shorten_label(label: str, max_chars: int = 28) -> str:
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1].rstrip() + "..."


def make_figure(
    examples: list[tuple[str, list[tuple[np.ndarray, str]]]],
    n_per_dataset: int,
    seed: int,
) -> plt.Figure:
    configure_style()

    n_rows = len(examples)
    fig_width = max(8.0, 1.85 * n_per_dataset + 1.2)
    fig_height = max(3.2, 1.95 * n_rows + 1.4)
    fig, axes = plt.subplots(
        n_rows,
        n_per_dataset,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=False,
    )

    fig.suptitle(
        "Ejemplos reales por dataset",
        fontsize=15,
        fontweight="bold",
        color=COLORS["dark_blue"],
        y=0.985,
    )
    fig.text(
        0.5,
        0.955,
        f"Muestreo aleatorio reproducible: N={n_per_dataset}, seed={seed}",
        ha="center",
        va="top",
        fontsize=9,
        color=COLORS["gray"],
    )

    for row_idx, (dataset, selected) in enumerate(examples):
        for col_idx in range(n_per_dataset):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color(COLORS["light_blue"])

            if col_idx < len(selected):
                image, superclass = selected[col_idx]
                ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
                ax.set_xlabel(shorten_label(superclass), labelpad=5, color="#222222")
            else:
                ax.set_facecolor("#F8F9FA")
                ax.text(
                    0.5,
                    0.5,
                    "Sin imagen\nvalida",
                    ha="center",
                    va="center",
                    color=COLORS["gray"],
                    fontsize=8,
                    transform=ax.transAxes,
                )

            if col_idx == 0:
                ax.set_ylabel(
                    dataset,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=34,
                    fontsize=10,
                    fontweight="bold",
                    color=COLORS["dark_blue"],
                )

    fig.text(0.01, 0.012, SOURCE_NOTE, ha="left", va="bottom", fontsize=8, color=COLORS["gray"])
    fig.subplots_adjust(left=0.12, right=0.985, top=0.92, bottom=0.065, hspace=0.38, wspace=0.08)
    return fig


def save_outputs(fig: plt.Figure, output_path: Path) -> None:
    if output_path.suffix.lower() != ".png":
        fail("--output debe terminar en .png para cumplir la exportacion obligatoria en PNG.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Figura PNG guardada en: {output_path}")
    print(f"Figura SVG guardada en: {svg_path}")


def main() -> None:
    args = parse_args()
    validate_inputs(args.metadata, args.n_per_dataset)
    df = load_metadata(args.metadata)
    examples = sample_valid_examples(df, args.metadata, args.n_per_dataset, args.seed)
    fig = make_figure(examples, args.n_per_dataset, args.seed)
    save_outputs(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
