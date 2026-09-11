#!/usr/bin/env python3
"""Grid comparativo de heterogeneidad entre datasets: 3 melanomas por dataset.

Filas (orden obligatorio): HAM10000, DERM12345, ISIC2019, PH2, Fitzpatrick17k.
Cada fila muestra 3 imagenes de melanoma (superclass == "melanoma") tomadas del
split especifico de ese dataset dentro del diseno experimental:

    HAM10000        split == "train"
    DERM12345       split == "train"
    ISIC2019        split == "test"
    PH2             split == "test_control"
    Fitzpatrick17k  split == "test_fairness"

Fuente: data/integrated/master_metadata_split_corrected.csv. Las imagenes se
resuelven a partir de la columna "path" del CSV maestro (ruta original de
adquisicion, p.ej. .../data/ogs/<dataset>/images/<archivo>), remapeando el
prefijo remoto de la maquina de origen (".../TFM/") a la raiz local del
repositorio. NO se usan las carpetas actuales de data/imagefolder ni
data/gen: cada ruta resuelta se verifica explicitamente contra data/ogs/.

No genera imagenes nuevas ni aplica IA generativa: solo compone una figura a
partir de archivos ya existentes. Las imagenes no se recortan, deforman ni
alteran (sin cambios de brillo/contraste/saturacion/color/nitidez); se ajustan
dentro de una celda cuadrada mediante padding blanco, conservando su aspect
ratio original.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps

DEFAULT_MASTER_CSV = Path("data/integrated/master_metadata_split_corrected.csv")
DEFAULT_OUTPUT = Path("reports/figures/final/fig_dataset_examples_grid.png")
DEFAULT_MANIFEST = Path("reports/results/dataset_grid_manifest.csv")
DEFAULT_SEED = 27
DEFAULT_NUM_PER_DATASET = 3

# Orden obligatorio de filas y filtro (dataset, split); todas se filtran ademas
# por superclass == "melanoma" (taxonomia consolidada del CSV maestro, valida
# tambien para PH2).
DATASET_ROWS = [
    ("HAM10000", "train"),
    ("DERM12345", "train"),
    ("ISIC2019", "test"),
    ("PH2", "test_control"),
    ("Fitzpatrick17k", "test_fairness"),
]

# Ruta reservada: si una ruta resuelta no cae bajo data/ogs/, indicaria que el
# CSV maestro apunta a una carpeta experimental (imagefolder/gen) en lugar de
# la fuente real de adquisicion.
REQUIRED_REAL_SUBTREE = Path("data/ogs")

MANIFEST_COLUMNS = [
    "dataset", "position_in_row", "filename", "source_path",
    "split", "superclass", "selection_mode", "seed",
]

CELL_PX = 512  # resolucion interna de cada celda (antes de componer la figura)
FIGURE_WIDTH_CM = 24.0
CELL_GAP = 0.03
BORDER_WIDTH_PT = 1.0
DPI = 300
LABEL_FONTSIZE = 13

CALIBRI_CANDIDATES = [
    Path("/Applications/Microsoft Excel.app/Contents/Resources/DFonts/Calibri.ttf"),
    Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts/Calibri.ttf"),
    Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/Calibri.ttf"),
    Path("/Applications/Microsoft Outlook.app/Contents/Resources/DFonts/Calibri.ttf"),
    Path("/System/Library/PrivateFrameworks/FontServices.framework/Versions/A/Resources/Fonts/ApplicationSupport/Carlito.ttc"),
]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def needs_decision(message: str) -> None:
    print("NEEDS_DECISION")
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "data").is_dir() and (candidate / "reports").is_dir():
            return candidate
    raise RuntimeError(f"Could not auto-locate repo root from {start}; pass --repo-root explicitly.")


def register_calibri() -> str:
    """Register a real Calibri font file if available on this machine; else fall back
    to a metric-compatible substitute (Carlito) or the default sans-serif family."""
    for candidate in CALIBRI_CANDIDATES:
        if candidate.exists():
            try:
                fm.fontManager.addfont(str(candidate))
                name = fm.FontProperties(fname=str(candidate)).get_name()
                print(f"[AUDIT] Fuente registrada: {name} ({candidate})")
                return name
            except Exception as exc:
                print(f"[WARN] No se pudo registrar la fuente '{candidate}': {exc}", file=sys.stderr)
    print("[WARN] No se encontro Calibri/Carlito en el sistema; se usara la fuente sans-serif por defecto.", file=sys.stderr)
    return "sans-serif"


def resolve_original_path(repo_root: Path, raw_path: str) -> Path:
    """Map an original (possibly remote-machine) path recorded in the master CSV to
    this machine's local repo path, by locating the "/data/" anchor and rejoining
    it under repo_root. E.g. "/home/marsi/TFM/data/ogs/X/img.jpg" ->
    "<repo_root>/data/ogs/X/img.jpg"."""
    p = str(raw_path)
    anchor = "/data/"
    idx = p.find(anchor)
    if idx == -1:
        rel = p.lstrip("/")
    else:
        rel = p[idx + 1:]  # keep leading "data/..."
    return repo_root / rel


def resolve_dataset_candidates(repo_root: Path, df: pd.DataFrame, dataset: str, split: str):
    """Melanoma pool for one row: dataset == dataset, split == split,
    superclass == 'melanoma'. Resolved via the CSV's own 'path' column, keyed by
    basename (as it appears on disk, with extension)."""
    sub = df[(df["dataset"] == dataset) & (df["split"] == split) & (df["superclass"] == "melanoma")]

    non_melanoma_flag = sub[sub["is_melanoma"] != 1]
    if len(non_melanoma_flag) > 0:
        fail(
            f"[{dataset}] {len(non_melanoma_flag)} registros con superclass=='melanoma' "
            f"pero is_melanoma != 1; abortando por seguridad."
        )

    basenames = [Path(str(p)).name for p in sub["path"]]
    dup_mask = pd.Series(basenames).duplicated(keep=False)
    if dup_mask.any():
        dups = sorted(set(pd.Series(basenames)[dup_mask]))
        fail(f"[{dataset}] basenames duplicados en el subconjunto filtrado: {dups[:10]}")

    resolved = {}
    missing = []
    contaminated = []
    for basename, raw_path in zip(basenames, sub["path"]):
        local = resolve_original_path(repo_root, raw_path)
        try:
            rel_to_root = local.resolve().relative_to(repo_root.resolve())
        except ValueError:
            rel_to_root = None
        under_real_source = rel_to_root is not None and (
            rel_to_root == REQUIRED_REAL_SUBTREE or REQUIRED_REAL_SUBTREE in rel_to_root.parents
        )
        if not under_real_source:
            contaminated.append(basename)
            continue
        if local.exists():
            resolved[basename] = local
        else:
            missing.append(basename)

    if contaminated:
        fail(
            f"[{dataset}] {len(contaminated)} registros resuelven fuera de {REQUIRED_REAL_SUBTREE}/ "
            f"(fuente real de adquisicion); no se puede depender de esas rutas. "
            f"Ejemplos: {contaminated[:5]}"
        )

    return sub, basenames, resolved, missing


def load_manual_config(filenames_config: Path) -> dict:
    """CSV con columnas 'dataset' y 'filename'. Devuelve {dataset: [filename, ...]}
    preservando el orden de aparicion en el fichero."""
    if not filenames_config.exists():
        fail(f"No se encontro el fichero de seleccion manual: {filenames_config}")

    df = pd.read_csv(filenames_config, dtype=str)
    cols_lower = {c.strip().lower(): c for c in df.columns}
    if "dataset" not in cols_lower or "filename" not in cols_lower:
        fail(f"'{filenames_config}' debe contener columnas 'dataset' y 'filename'.")

    by_dataset: dict = {}
    for _, row in df.iterrows():
        dataset = str(row[cols_lower["dataset"]]).strip()
        filename = str(row[cols_lower["filename"]]).strip()
        if not dataset or not filename:
            continue
        by_dataset.setdefault(dataset, []).append(filename)
    return by_dataset


def select_manual(dataset: str, filenames: list, resolved: dict, num: int) -> list:
    if len(filenames) != num:
        fail(
            f"[{dataset}] la seleccion manual contiene {len(filenames)} filenames "
            f"pero --num-per-dataset es {num}."
        )

    dup_mask = pd.Series(filenames).duplicated(keep=False)
    if dup_mask.any():
        dups = sorted(set(pd.Series(filenames)[dup_mask]))
        fail(f"[{dataset}] filenames duplicados en seleccion manual: {dups}")

    not_valid = [f for f in filenames if f not in resolved]
    if not_valid:
        fail(
            f"[{dataset}] los siguientes filenames no pertenecen al subconjunto "
            f"dataset=='{dataset}' & superclass=='melanoma', o no son resolubles "
            f"fisicamente: {not_valid}"
        )

    return list(filenames)


def deterministic_sample(dataset: str, resolved: dict, num: int, seed: int) -> list:
    keys = sorted(resolved.keys())
    if num > len(keys):
        needs_decision(
            f"[{dataset}] se solicitaron {num} imagenes de melanoma resolubles pero "
            f"solo hay {len(keys)} disponibles bajo el filtro aplicado."
        )
    first = random.Random(seed).sample(keys, num)
    second = random.Random(seed).sample(keys, num)
    if first != second:
        fail(f"[{dataset}] la seleccion aleatoria no es reproducible para seed={seed}.")
    return first


def pad_to_square(path: Path, cell_px: int) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return ImageOps.pad(im, (cell_px, cell_px), method=Image.LANCZOS, color=(255, 255, 255), centering=(0.5, 0.5))


def build_figure(images_by_row: list, row_labels: list, font_name: str) -> plt.Figure:
    n_rows = len(images_by_row)
    n_cols = len(images_by_row[0])
    width_in = FIGURE_WIDTH_CM / 2.54
    height_in = width_in * (n_rows / n_cols) * 1.04

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_in, height_in), squeeze=False)
    fig.patch.set_facecolor("white")

    for row_idx, images in enumerate(images_by_row):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("white")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(BORDER_WIDTH_PT)
            ax.imshow(images[col_idx])
            ax.margins(0)

        axes[row_idx][0].text(
            -0.12,
            0.5,
            row_labels[row_idx],
            transform=axes[row_idx][0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=LABEL_FONTSIZE,
            fontfamily=font_name,
            color="black",
        )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.995, bottom=0.005, wspace=CELL_GAP, hspace=CELL_GAP)
    return fig


def write_manifest(manifest_path: Path, rows: list) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    print(f"[OUTPUT] Manifest guardado en: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=None, help="Repo root (auto-detectado si se omite).")
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER_CSV)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla reproducible. Por defecto: 27.")
    parser.add_argument("--num-per-dataset", type=int, default=DEFAULT_NUM_PER_DATASET, help="Imagenes por dataset. Por defecto: 3.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--filenames-config",
        type=Path,
        default=None,
        help=(
            "CSV con columnas 'dataset' y 'filename' para seleccion manual. Los "
            "datasets presentes en el fichero usan exactamente esos filenames "
            "(verificados); los datasets ausentes se muestrean aleatoriamente."
        ),
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, p: Path) -> Path:
    return p if p.is_absolute() else repo_root / p


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path(__file__).resolve())

    master_csv = resolve_path(repo_root, args.master_csv)
    output_path = resolve_path(repo_root, args.output)
    manifest_path = resolve_path(repo_root, args.manifest)

    if not master_csv.exists():
        fail(f"No se encontro el CSV maestro: {master_csv}")

    df = pd.read_csv(master_csv, low_memory=False)
    print(f"[AUDIT] Fuente: {master_csv}")
    print(f"[AUDIT] Filas totales en el CSV maestro: {len(df)}")

    manual_by_dataset = load_manual_config(resolve_path(repo_root, args.filenames_config)) if args.filenames_config else {}

    print("[AUDIT] Valores reales de 'dataset':", sorted(df["dataset"].dropna().unique().tolist()))
    print("[AUDIT] Valores reales de 'split':", sorted(df["split"].dropna().unique().tolist()))
    print("[AUDIT] Valores reales de 'superclass':", sorted(df["superclass"].dropna().unique().tolist()))

    manifest_rows = []
    images_by_row = []
    row_labels = []

    for dataset, split in DATASET_ROWS:
        print(f"[AUDIT] === {dataset} (split=='{split}', superclass=='melanoma') ===")
        if dataset not in set(df["dataset"].dropna().unique()):
            needs_decision(f"[{dataset}] no aparece en la columna 'dataset' del CSV maestro.")

        sub, basenames, resolved, missing = resolve_dataset_candidates(repo_root, df, dataset, split)
        available = len(basenames)
        resolvable = len(resolved)
        print(f"[AUDIT] Disponibles: {available}  Resolubles: {resolvable}  Faltantes: {len(missing)}")
        if missing:
            print(f"[AUDIT][WARN] Ejemplos no resolubles: {missing[:5]}")

        if resolvable < args.num_per_dataset:
            needs_decision(
                f"[{dataset}] solo hay {resolvable} melanomas resolubles bajo "
                f"dataset=='{dataset}' & split=='{split}' & superclass=='melanoma', "
                f"pero se requieren {args.num_per_dataset}."
            )

        if dataset in manual_by_dataset:
            selected = select_manual(dataset, manual_by_dataset[dataset], resolved, args.num_per_dataset)
            mode = "manual"
        else:
            selected = deterministic_sample(dataset, resolved, args.num_per_dataset, args.seed)
            mode = "random"

        if len(set(selected)) != len(selected):
            fail(f"[{dataset}] duplicados en la seleccion final de la fila.")
        for filename in selected:
            if not resolved[filename].exists():
                fail(f"[{dataset}] el archivo seleccionado no existe fisicamente: {resolved[filename]}")

        print(f"[AUDIT] Seleccion {dataset} ({mode}, seed={args.seed}):")
        for idx, filename in enumerate(selected, start=1):
            print(f"  {idx}  {filename}")

        for position, filename in enumerate(selected, start=1):
            manifest_rows.append({
                "dataset": dataset,
                "position_in_row": position,
                "filename": filename,
                "source_path": str(resolved[filename]),
                "split": split,
                "superclass": "melanoma",
                "selection_mode": mode,
                "seed": args.seed,
            })

        images_by_row.append([pad_to_square(resolved[f], CELL_PX) for f in selected])
        row_labels.append(dataset)

    if len(manifest_rows) != len(DATASET_ROWS) * args.num_per_dataset:
        fail(
            f"Se esperaban {len(DATASET_ROWS) * args.num_per_dataset} filas en el manifest "
            f"pero se generaron {len(manifest_rows)}."
        )

    font_name = register_calibri()

    fig = build_figure(images_by_row, row_labels, font_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[OUTPUT] Figura guardada en: {output_path}")

    write_manifest(manifest_path, manifest_rows)


if __name__ == "__main__":
    main()
