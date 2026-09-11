#!/usr/bin/env python3
"""Plot blur score distribution and summarize the 5th percentile threshold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METADATA = Path("data/other/no_fitz/master_metadata.csv")
DEFAULT_OUTPUT = Path("reports/figures/fig_blur_percentil5.png")
DEFAULT_SUMMARY_OUTPUT = Path("reports/figures/blur_distribution_summary.csv")
DEFAULT_UPDATED_METADATA = Path("reports/figures/master_metadata_with_blur_score.csv")
BLUR_COLUMN = "blur_score"
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
            "Lee master_metadata.csv, usa una columna de blur si existe o calcula "
            "blur_score con varianza del Laplaciano, y genera un histograma con el percentil 5."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Ruta al CSV de metadatos. Por defecto: {DEFAULT_METADATA}",
    )
    parser.add_argument(
        "--blur-column",
        default=BLUR_COLUMN,
        help=f"Nombre de la columna de blur ya calculada. Por defecto: {BLUR_COLUMN}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ruta del PNG de salida. Por defecto: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help=f"Ruta del CSV resumen. Por defecto: {DEFAULT_SUMMARY_OUTPUT}",
    )
    parser.add_argument(
        "--save-updated-metadata",
        action="store_true",
        help="Si se calcula blur desde imagenes, guarda una copia del metadata con la nueva columna.",
    )
    parser.add_argument(
        "--updated-metadata-output",
        type=Path,
        default=DEFAULT_UPDATED_METADATA,
        help=f"Ruta de la copia actualizada. Por defecto: {DEFAULT_UPDATED_METADATA}",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Numero de bins del histograma. Por defecto: 40.",
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
    if args.bins < 5:
        fail("--bins debe ser mayor o igual que 5.")
    if args.save_updated_metadata and args.updated_metadata_output.resolve() == args.metadata.resolve():
        fail("--updated-metadata-output no puede sobrescribir el CSV original.")


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(metadata_path, low_memory=False)
    except Exception as exc:
        fail(f"No se pudo leer el CSV '{metadata_path}': {exc}")
    if df.empty:
        fail("El CSV de metadatos esta vacio.")
    return df


def resolve_image_path(path_value: str, metadata_path: Path) -> Path:
    image_path = Path(path_value)
    if image_path.is_absolute():
        return image_path

    repo_relative = Path.cwd() / image_path
    if repo_relative.exists():
        return repo_relative

    return metadata_path.parent / image_path


def calculate_blur_scores(df: pd.DataFrame, metadata_path: Path, blur_column: str) -> pd.DataFrame:
    if "path" not in df.columns:
        fail(
            f"No existe la columna '{blur_column}' y tampoco existe la columna 'path' "
            "para calcular blur desde imagenes."
        )

    try:
        import cv2
    except Exception as exc:
        fail(
            "No existe una columna de blur calculada y no se pudo importar OpenCV "
            f"(cv2) para calcularla: {exc}"
        )

    updated = df.copy()
    scores: list[float | None] = []
    total = len(updated)

    for idx, path_value in enumerate(updated["path"], start=1):
        if pd.isna(path_value) or str(path_value).strip() == "":
            scores.append(None)
            continue

        image_path = resolve_image_path(str(path_value).strip(), metadata_path)
        if not image_path.exists() or not image_path.is_file():
            scores.append(None)
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            scores.append(None)
            continue

        score = float(cv2.Laplacian(image, cv2.CV_64F).var())
        scores.append(score)

        if idx % 5000 == 0:
            print(f"Calculadas {idx}/{total} imagenes...", file=sys.stderr, flush=True)

    updated[blur_column] = scores
    valid_count = updated[blur_column].notna().sum()
    if valid_count == 0:
        fail("No se pudo calcular blur para ninguna imagen valida.")

    missing_count = total - valid_count
    if missing_count:
        print(
            f"AVISO: {missing_count} filas no tienen blur calculado por ruta inexistente, "
            "imagen no legible o path vacio.",
            file=sys.stderr,
        )

    return updated


def get_blur_scores(df: pd.DataFrame, metadata_path: Path, blur_column: str) -> tuple[pd.DataFrame, bool]:
    if blur_column in df.columns:
        updated = df.copy()
        updated[blur_column] = pd.to_numeric(updated[blur_column], errors="coerce")
        if updated[blur_column].notna().sum() == 0:
            fail(f"La columna '{blur_column}' existe, pero no contiene valores numericos validos.")
        return updated, False

    print(
        f"AVISO: no existe la columna '{blur_column}'. Se calculara desde imagenes usando 'path'.",
        file=sys.stderr,
    )
    return calculate_blur_scores(df, metadata_path, blur_column), True


def build_summary(scores: pd.Series) -> pd.DataFrame:
    clean_scores = scores.dropna()
    p5 = clean_scores.quantile(0.05)
    summary = {
        "threshold": p5,
        "mean": clean_scores.mean(),
        "median": clean_scores.median(),
        "p5": p5,
        "p25": clean_scores.quantile(0.25),
        "p75": clean_scores.quantile(0.75),
        "discardable_images": int((clean_scores <= p5).sum()),
        "valid_images": int(clean_scores.shape[0]),
        "missing_blur_scores": int(scores.isna().sum()),
    }
    return pd.DataFrame([summary])


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


def make_plot(scores: pd.Series, p5: float, bins: int) -> plt.Figure:
    configure_style()
    clean_scores = scores.dropna()

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.hist(
        clean_scores,
        bins=bins,
        color=COLORS["dark_blue"],
        edgecolor="white",
        linewidth=0.7,
        alpha=0.92,
    )
    ax.axvline(
        p5,
        color="#C45A25",
        linestyle="--",
        linewidth=2.0,
        label=f"Percentil 5 = {p5:.2f}",
    )

    ax.set_title("Distribucion del blur score", fontweight="bold", color=COLORS["dark_blue"], pad=18)
    ax.text(
        0.0,
        1.02,
        "Varianza del Laplaciano; valores bajos indican mayor desenfoque",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=COLORS["gray"],
    )
    ax.set_xlabel("Blur score")
    ax.set_ylabel("Numero de imagenes")
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.01, 0.012, SOURCE_NOTE, ha="left", va="bottom", fontsize=8, color=COLORS["gray"])
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    return fig


def save_outputs(
    fig: plt.Figure,
    summary: pd.DataFrame,
    df: pd.DataFrame,
    calculated_blur: bool,
    args: argparse.Namespace,
) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(args.summary_output, index=False)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    svg_output = args.output.with_suffix(".svg")
    fig.savefig(svg_output, bbox_inches="tight")

    print(f"Resumen CSV guardado en: {args.summary_output}")
    print(f"Figura PNG guardada en: {args.output}")
    print(f"Figura SVG guardada en: {svg_output}")

    if calculated_blur and args.save_updated_metadata:
        args.updated_metadata_output.parent.mkdir(parents=True, exist_ok=True)
        if args.updated_metadata_output.resolve() == args.metadata.resolve():
            fail("La copia actualizada no puede sobrescribir el CSV original.")
        df.to_csv(args.updated_metadata_output, index=False)
        print(f"Copia de metadata con blur guardada en: {args.updated_metadata_output}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    df = load_metadata(args.metadata)
    df, calculated_blur = get_blur_scores(df, args.metadata, args.blur_column)

    scores = pd.to_numeric(df[args.blur_column], errors="coerce")
    summary = build_summary(scores)
    p5 = float(summary.loc[0, "p5"])
    fig = make_plot(scores, p5, args.bins)
    save_outputs(fig, summary, df, calculated_blur, args)
    plt.close(fig)


if __name__ == "__main__":
    main()
