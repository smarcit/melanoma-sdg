#!/usr/bin/env python3
"""Grid comparativo: ejemplos aceptados vs. rechazados por el filtro semantico.

Grupo accepted:
    Fuente: data/synthetic_melanoma_accepted.csv (columna "path")
Grupo rejected:
    Fuente: data/synthetic_melanoma_rejected.csv (columna "path")

Ambos CSV son salida del filtro semantico basado en clasificador aplicado
tras los controles basicos de calidad y deduplicacion. El campo "path" de
estos CSV puede estar obsoleto (apunta a una maquina/estructura remota), por
lo que cada imagen se resuelve por basename contra
data/gen/SD_E2_HOPE/v1_8000/.

La distincion accepted/rejected corresponde exclusivamente al criterio del
filtro semantico implementado en el proyecto; no implica ninguna valoracion
clinica sobre la imagen.

No genera imagenes nuevas ni aplica IA generativa: solo compone una figura a
partir de archivos ya existentes. Las imagenes no se recortan, deforman ni
alteran (sin cambios de brillo/contraste/saturacion/color/nitidez).
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
from PIL import Image

DEFAULT_ACCEPTED_CSV = Path("data/synthetic_melanoma_accepted.csv")
DEFAULT_REJECTED_CSV = Path("data/synthetic_melanoma_rejected.csv")
DEFAULT_RESOLVE_DIR = Path("data/gen/SD_E2_HOPE/v1_8000")
DEFAULT_OUTPUT = Path("reports/figures/final/fig_semantic_filter_grid.png")
DEFAULT_MANIFEST = Path("reports/results/semantic_filter_grid_manifest.csv")
DEFAULT_SEED = 27
DEFAULT_NUM_PER_GROUP = 6

EXPECTED_ACCEPTED = 2043
EXPECTED_REJECTED = 142

ACCEPTED_GROUP = "accepted"
REJECTED_GROUP = "rejected"
ROW_LABELS = {ACCEPTED_GROUP: "Aceptadas", REJECTED_GROUP: "Rechazadas"}
MANIFEST_COLUMNS = ["group", "position", "filename", "source_path", "selection_mode", "seed"]

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


def load_basenames(csv_path: Path, group_label: str) -> list[str]:
    if not csv_path.exists():
        fail(f"[{group_label}] No se encontro el CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        fail(f"[{group_label}] El CSV '{csv_path}' debe contener una columna 'path'.")

    basenames = [Path(str(p)).name for p in df["path"]]
    dup_series = pd.Series(basenames)
    dup_mask = dup_series.duplicated(keep=False)
    if dup_mask.any():
        dups = sorted(set(dup_series[dup_mask]))
        fail(f"[{group_label}] Se han detectado {len(dups)} basenames duplicados en '{csv_path}': {dups[:10]}")

    print(f"[AUDIT] [{group_label}] Fuente: {csv_path}")
    print(f"[AUDIT] [{group_label}] Filas: {len(basenames)}")
    return basenames


def resolve_group(basenames: list[str], resolve_dir: Path, group_label: str) -> tuple[dict, list]:
    index = {}
    missing = []
    for name in basenames:
        candidate = resolve_dir / name
        if candidate.is_file():
            index[name] = candidate
        else:
            missing.append(name)

    print(f"[AUDIT] [{group_label}] Directorio de resolucion: {resolve_dir}")
    print(f"[AUDIT] [{group_label}] Resolubles: {len(index)} / {len(basenames)}")
    if missing:
        print(f"[AUDIT][WARN] [{group_label}] {len(missing)} no resolubles (no bloquea si no entran en la seleccion): {missing[:10]}")
    return index, missing


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


def select_manual(filenames_file: Path, resolved: dict, num: int, group_label: str) -> list[str]:
    selected = load_manual_filenames(filenames_file)

    if len(selected) != num:
        fail(
            f"[{group_label}] '{filenames_file}' contiene {len(selected)} filenames "
            f"pero --num-per-group es {num}."
        )

    dup_mask = pd.Series(selected).duplicated(keep=False)
    if dup_mask.any():
        dups = sorted(set(pd.Series(selected)[dup_mask]))
        fail(f"[{group_label}] filenames duplicados en seleccion manual: {dups}")

    not_valid = [f for f in selected if f not in resolved]
    if not_valid:
        fail(
            f"[{group_label}] los siguientes filenames no pertenecen al conjunto {group_label} "
            f"o no son resolubles fisicamente: {not_valid}"
        )

    return selected


def deterministic_sample(resolved: dict, num: int, seed: int, group_label: str) -> list[str]:
    keys = sorted(resolved.keys())
    if num > len(keys):
        fail(f"[{group_label}] Se solicitaron {num} imagenes pero solo hay {len(keys)} resolubles.")
    return random.Random(seed).sample(keys, num)


def select_group(group_label: str, resolved: dict, num: int, seed: int, manual_file: Path | None) -> tuple[list[str], str]:
    if manual_file is not None:
        selected = select_manual(manual_file, resolved, num, group_label)
        mode = "manual"
    else:
        selected = deterministic_sample(resolved, num, seed, group_label)
        repeat = deterministic_sample(resolved, num, seed, group_label)
        if selected != repeat:
            fail(f"[{group_label}] la seleccion aleatoria no es reproducible para seed={seed}.")
        mode = "random"
    return selected, mode


def load_image(path: Path) -> "Image.Image":
    with Image.open(path) as im:
        im.load()
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im.copy()


def build_figure(accepted_images: list, rejected_images: list, font_name: str) -> plt.Figure:
    cols = len(accepted_images)
    width_in = FIGURE_WIDTH_CM / 2.54
    height_in = width_in * (2 / cols) * 1.08  # margen extra vertical para los rotulos de fila

    fig, axes = plt.subplots(2, cols, figsize=(width_in, height_in), squeeze=False)
    fig.patch.set_facecolor("white")

    for row_idx, (images, group) in enumerate([(accepted_images, ACCEPTED_GROUP), (rejected_images, REJECTED_GROUP)]):
        for col_idx in range(cols):
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
            ROW_LABELS[group],
            transform=axes[row_idx][0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=LABEL_FONTSIZE,
            fontfamily=font_name,
            color="black",
        )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.995, bottom=0.01, wspace=CELL_GAP, hspace=CELL_GAP)
    return fig


def write_manifest(
    manifest_path: Path,
    accepted_selected: list[str],
    accepted_resolved: dict,
    accepted_mode: str,
    rejected_selected: list[str],
    rejected_resolved: dict,
    rejected_mode: str,
    seed: int,
) -> None:
    records = []
    for idx, filename in enumerate(accepted_selected, start=1):
        records.append({
            "group": ACCEPTED_GROUP,
            "position": idx,
            "filename": filename,
            "source_path": str(accepted_resolved[filename]),
            "selection_mode": accepted_mode,
            "seed": seed,
        })
    offset = len(accepted_selected)
    for idx, filename in enumerate(rejected_selected, start=1):
        records.append({
            "group": REJECTED_GROUP,
            "position": offset + idx,
            "filename": filename,
            "source_path": str(rejected_resolved[filename]),
            "selection_mode": rejected_mode,
            "seed": seed,
        })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    print(f"[OUTPUT] Manifest guardado en: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=None, help="Repo root (auto-detectado si se omite).")
    parser.add_argument("--accepted-csv", type=Path, default=DEFAULT_ACCEPTED_CSV)
    parser.add_argument("--rejected-csv", type=Path, default=DEFAULT_REJECTED_CSV)
    parser.add_argument("--resolve-dir", type=Path, default=DEFAULT_RESOLVE_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla reproducible. Por defecto: 27.")
    parser.add_argument("--num-per-group", type=int, default=DEFAULT_NUM_PER_GROUP, help="Imagenes por grupo. Por defecto: 6.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--accepted-filenames-file",
        type=Path,
        default=None,
        help="TXT (una linea por filename) o CSV (columna 'filename', o la primera) con seleccion manual de aceptadas.",
    )
    parser.add_argument(
        "--rejected-filenames-file",
        type=Path,
        default=None,
        help="TXT (una linea por filename) o CSV (columna 'filename', o la primera) con seleccion manual de rechazadas.",
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, p: Path) -> Path:
    return p if p.is_absolute() else repo_root / p


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path(__file__).resolve())

    accepted_csv = resolve_path(repo_root, args.accepted_csv)
    rejected_csv = resolve_path(repo_root, args.rejected_csv)
    resolve_dir = resolve_path(repo_root, args.resolve_dir)
    output_path = resolve_path(repo_root, args.output)
    manifest_path = resolve_path(repo_root, args.manifest)

    if not resolve_dir.is_dir():
        fail(f"No se encontro el directorio de resolucion: {resolve_dir}")

    print("[AUDIT] === Grupo accepted ===")
    accepted_basenames = load_basenames(accepted_csv, ACCEPTED_GROUP)
    accepted_resolved, accepted_missing = resolve_group(accepted_basenames, resolve_dir, ACCEPTED_GROUP)
    if len(accepted_basenames) != EXPECTED_ACCEPTED:
        print(
            f"[AUDIT][WARN] Se esperaban {EXPECTED_ACCEPTED} filas accepted pero el CSV contiene "
            f"{len(accepted_basenames)}.",
            file=sys.stderr,
        )

    print("[AUDIT] === Grupo rejected ===")
    rejected_basenames = load_basenames(rejected_csv, REJECTED_GROUP)
    rejected_resolved, rejected_missing = resolve_group(rejected_basenames, resolve_dir, REJECTED_GROUP)
    if len(rejected_basenames) != EXPECTED_REJECTED:
        print(
            f"[AUDIT][WARN] Se esperaban {EXPECTED_REJECTED} filas rejected pero el CSV contiene "
            f"{len(rejected_basenames)}.",
            file=sys.stderr,
        )

    accepted_set = set(accepted_basenames)
    rejected_set = set(rejected_basenames)
    overlap = sorted(accepted_set & rejected_set)
    if overlap:
        print("NEEDS_DECISION")
        fail(
            f"Se han detectado {len(overlap)} filenames presentes simultaneamente en accepted y "
            f"rejected: {overlap[:10]}"
        )

    accepted_selected, accepted_mode = select_group(
        ACCEPTED_GROUP, accepted_resolved, args.num_per_group, args.seed, args.accepted_filenames_file
    )
    rejected_selected, rejected_mode = select_group(
        REJECTED_GROUP, rejected_resolved, args.num_per_group, args.seed, args.rejected_filenames_file
    )

    if len(set(accepted_selected)) != len(accepted_selected):
        fail(f"[{ACCEPTED_GROUP}] duplicados en la seleccion final.")
    if len(set(rejected_selected)) != len(rejected_selected):
        fail(f"[{REJECTED_GROUP}] duplicados en la seleccion final.")
    for filename in accepted_selected:
        if not accepted_resolved[filename].exists():
            fail(f"[{ACCEPTED_GROUP}] el archivo seleccionado no existe fisicamente: {accepted_resolved[filename]}")
    for filename in rejected_selected:
        if not rejected_resolved[filename].exists():
            fail(f"[{REJECTED_GROUP}] el archivo seleccionado no existe fisicamente: {rejected_resolved[filename]}")

    print(f"[AUDIT] Seleccion accepted ({accepted_mode}, seed={args.seed}):")
    for idx, filename in enumerate(accepted_selected, start=1):
        print(f"  {idx}  {filename}")
    print(f"[AUDIT] Seleccion rejected ({rejected_mode}, seed={args.seed}):")
    for idx, filename in enumerate(rejected_selected, start=1):
        print(f"  {idx}  {filename}")

    font_name = register_calibri()

    accepted_images = [load_image(accepted_resolved[f]) for f in accepted_selected]
    rejected_images = [load_image(rejected_resolved[f]) for f in rejected_selected]

    fig = build_figure(accepted_images, rejected_images, font_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[OUTPUT] Figura guardada en: {output_path}")

    write_manifest(
        manifest_path,
        accepted_selected, accepted_resolved, accepted_mode,
        rejected_selected, rejected_resolved, rejected_mode,
        args.seed,
    )


if __name__ == "__main__":
    main()
