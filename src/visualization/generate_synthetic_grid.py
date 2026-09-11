#!/usr/bin/env python3
"""Generate a clean academic grid figure of accepted synthetic melanoma images.

Resolves images by basename against a generation directory because the
`path` column recorded in the accepted CSV is stale for a relevant share of
rows (it still points at the old imagefolder layout).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_ACCEPTED_CSV = Path("data/synthetic_melanoma_accepted.csv")
DEFAULT_RESOLVE_DIR = Path("data/gen/SD_E2_HOPE/v1_8000")
DEFAULT_OUTPUT = Path("reports/figures/final/fig_synthetic_accepted_grid.png")
DEFAULT_MANIFEST = Path("reports/results/synthetic_accepted_grid_manifest.csv")
DEFAULT_SEED = 27
DEFAULT_NUM_IMAGES = 12
DEFAULT_ROWS = 3
DEFAULT_COLS = 4

FIGURE_WIDTH_CM = 16.0
CELL_GAP = 0.025
BORDER_WIDTH_PT = 1.0
DPI = 300


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye un grid academico (sin texto ni titulos) con ejemplos "
            "reales del conjunto de imagenes sinteticas de melanoma aceptadas "
            "tras el filtro semantico."
        )
    )
    parser.add_argument(
        "--accepted-csv",
        type=Path,
        default=DEFAULT_ACCEPTED_CSV,
        help=f"CSV de imagenes aceptadas. Por defecto: {DEFAULT_ACCEPTED_CSV}",
    )
    parser.add_argument(
        "--resolve-dir",
        type=Path,
        default=DEFAULT_RESOLVE_DIR,
        help=(
            "Directorio donde resolver los ficheros por basename (el campo "
            f"'path' del CSV puede estar obsoleto). Por defecto: {DEFAULT_RESOLVE_DIR}"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla reproducible. Por defecto: 27.")
    parser.add_argument(
        "--num-images", type=int, default=DEFAULT_NUM_IMAGES, help="Numero total de imagenes en el grid. Por defecto: 12."
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Filas del grid. Por defecto: 3.")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="Columnas del grid. Por defecto: 4.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ruta del PNG de salida. Por defecto: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Ruta del CSV de trazabilidad. Por defecto: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--filenames-file",
        type=Path,
        default=None,
        help=(
            "TXT (una linea por filename) o CSV (columna 'filename', o la "
            "primera columna) con una seleccion manual. Si se proporciona, "
            "desactiva el muestreo aleatorio y usa exactamente esos ficheros."
        ),
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="No generar la version PDF adicional junto al PNG.",
    )
    return parser.parse_args()


def load_accepted_basenames(accepted_csv: Path) -> list[str]:
    if not accepted_csv.exists():
        fail(f"No se encontro el CSV de aceptadas: {accepted_csv}")

    df = pd.read_csv(accepted_csv)
    if "path" not in df.columns:
        fail(f"El CSV de aceptadas '{accepted_csv}' debe contener una columna 'path'.")

    basenames = [Path(str(p)).name for p in df["path"]]
    duplicated = pd.Series(basenames)
    dup_mask = duplicated.duplicated(keep=False)
    if dup_mask.any():
        dups = sorted(set(duplicated[dup_mask]))
        fail(f"Se han detectado {len(dups)} basenames duplicados en el CSV de aceptadas: {dups[:10]}")

    print(f"[AUDIT] CSV aceptadas: {accepted_csv}")
    print(f"[AUDIT] Filas aceptadas: {len(basenames)}")
    return basenames


def audit_resolution(basenames: list[str], resolve_dir: Path) -> None:
    if not resolve_dir.exists() or not resolve_dir.is_dir():
        fail(f"El directorio de resolucion no existe: {resolve_dir}")

    missing = [b for b in basenames if not (resolve_dir / b).is_file()]
    resolvable = len(basenames) - len(missing)
    print(f"[AUDIT] Directorio de resolucion: {resolve_dir}")
    print(f"[AUDIT] Imagenes aceptadas resolubles: {resolvable} / {len(basenames)}")
    if missing:
        print(f"[AUDIT][WARN] {len(missing)} imagenes aceptadas no se pudieron resolver (no bloquea si no entran en la seleccion): {missing[:10]}")


def load_manual_filenames(filenames_file: Path) -> list[str]:
    if not filenames_file.exists():
        fail(f"No se encontro el fichero de seleccion manual: {filenames_file}")

    if filenames_file.suffix.lower() == ".csv":
        df = pd.read_csv(filenames_file)
        column = next((c for c in df.columns if c.strip().lower() == "filename"), df.columns[0])
        values = [str(v).strip() for v in df[column].tolist()]
    else:
        values = [line.strip() for line in filenames_file.read_text(encoding="utf-8").splitlines()]

    return [v for v in values if v]


def select_manual(filenames_file: Path, accepted_set: set[str], resolve_dir: Path, num_images: int) -> list[str]:
    selected = load_manual_filenames(filenames_file)

    if len(selected) != num_images:
        fail(
            f"El fichero de seleccion manual '{filenames_file}' contiene {len(selected)} "
            f"filenames pero --num-images es {num_images}."
        )

    dup_mask = pd.Series(selected).duplicated(keep=False)
    if dup_mask.any():
        dups = sorted(set(pd.Series(selected)[dup_mask]))
        fail(f"El fichero de seleccion manual contiene filenames duplicados: {dups}")

    not_accepted = [f for f in selected if f not in accepted_set]
    if not_accepted:
        fail(f"Los siguientes filenames no pertenecen al conjunto de aceptadas: {not_accepted}")

    missing = [f for f in selected if not (resolve_dir / f).is_file()]
    if missing:
        fail(f"Los siguientes filenames no existen fisicamente en '{resolve_dir}': {missing}")

    return selected


def select_random(basenames: list[str], num_images: int, seed: int) -> list[str]:
    if num_images > len(basenames):
        fail(f"--num-images ({num_images}) no puede superar el numero de imagenes aceptadas ({len(basenames)}).")

    def draw() -> list[str]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(basenames), size=num_images, replace=False)
        return [basenames[i] for i in idx]

    selected = draw()
    if selected != draw():
        fail("La seleccion aleatoria no es reproducible para la misma seed (no deberia ocurrir).")

    print(f"[AUDIT] Seleccion aleatoria reproducible confirmada para seed={seed}.")
    return selected


def load_images(selected: list[str], resolve_dir: Path) -> list[np.ndarray]:
    images = []
    for filename in selected:
        image_path = resolve_dir / filename
        try:
            with Image.open(image_path) as im:
                im.load()
                if im.mode != "RGB":
                    im = im.convert("RGB")
                images.append(np.asarray(im))
        except Exception as exc:
            fail(f"No se pudo abrir la imagen '{image_path}': {exc}")
    return images


def build_grid_figure(images: list[np.ndarray], rows: int, cols: int) -> plt.Figure:
    width_in = FIGURE_WIDTH_CM / 2.54
    height_in = width_in * (rows / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(width_in, height_in), squeeze=False)
    fig.patch.set_facecolor("white")

    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(BORDER_WIDTH_PT)

        if idx < len(images):
            ax.imshow(images[idx])
        ax.margins(0)

    fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002, wspace=CELL_GAP, hspace=CELL_GAP)
    return fig


def save_outputs(fig: plt.Figure, output_path: Path, save_pdf: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, facecolor="white")
    print(f"[OUTPUT] Figura PNG guardada en: {output_path}")

    if save_pdf:
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, facecolor="white")
        print(f"[OUTPUT] Figura PDF guardada en: {pdf_path}")


def write_manifest(
    manifest_path: Path,
    selected: list[str],
    resolve_dir: Path,
    rows: int,
    cols: int,
    selection_mode: str,
    seed: int,
) -> None:
    records = []
    for idx, filename in enumerate(selected):
        row = idx // cols + 1
        col = idx % cols + 1
        records.append(
            {
                "position": idx + 1,
                "row": row,
                "column": col,
                "filename": filename,
                "source_path": str(resolve_dir / filename),
                "selection_mode": selection_mode,
                "seed": seed,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(manifest_path, index=False)
    print(f"[OUTPUT] Manifest guardado en: {manifest_path}")


def main() -> None:
    args = parse_args()

    if args.rows * args.cols != args.num_images:
        fail(
            f"--rows x --cols ({args.rows}x{args.cols}={args.rows * args.cols}) debe ser "
            f"igual a --num-images ({args.num_images})."
        )

    basenames = load_accepted_basenames(args.accepted_csv)
    audit_resolution(basenames, args.resolve_dir)
    accepted_set = set(basenames)

    if args.filenames_file is not None:
        selected = select_manual(args.filenames_file, accepted_set, args.resolve_dir, args.num_images)
        selection_mode = "manual"
    else:
        selected = select_random(basenames, args.num_images, args.seed)
        selection_mode = "random"

    print("[AUDIT] Filenames seleccionados:")
    for idx, filename in enumerate(selected, start=1):
        print(f"  {idx:2d}  {filename}")

    images = load_images(selected, args.resolve_dir)
    fig = build_grid_figure(images, args.rows, args.cols)
    save_outputs(fig, args.output, save_pdf=not args.no_pdf)
    plt.close(fig)

    write_manifest(args.manifest, selected, args.resolve_dir, args.rows, args.cols, selection_mode, args.seed)


if __name__ == "__main__":
    main()
