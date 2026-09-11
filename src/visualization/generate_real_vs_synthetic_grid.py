#!/usr/bin/env python3
"""Grid comparativo simple: melanomas reales (train) vs. sinteticos (E4_SYNTH).

Grupo real:
    split == "train", superclass == "melanoma"
    Fuente: data/integrated/master_metadata_split_corrected.csv
    Resuelto a partir de la columna "path" del CSV maestro (ruta original de
    adquisicion, p.ej. .../data/ogs/<dataset>/images/<archivo>), remapeando el
    prefijo remoto de la maquina de origen (".../TFM/") a la raiz local del
    repositorio. NO se utiliza data/imagefolder/train como fuente: esa carpeta
    contiene actualmente imagenes sinteticas anadidas para E4_SYNTH (auditoria
    previa), por lo que serviria un pool contaminado si se usara directamente.

    Validacion "real-only": el subconjunto train+melanoma del CSV maestro solo
    contiene registros de datasets de adquisicion originales (HAM10000,
    DERM12345), con is_melanoma == 1, y sus rutas resueltas caen bajo
    data/ogs/... En tiempo de ejecucion se verifica ademas que ninguna ruta
    resuelta caiga bajo data/imagefolder/ ni data/gen/ (arboles reservados a
    imagefolder mezclado y a generacion sintetica, respectivamente).

Grupo sintetico:
    reports/results/synthetic_final_traceability.csv, used_in_E4_SYNTH == 1
    Resuelto por basename contra data/gen/SD_E2_HOPE/v1_8000 (misma logica que
    resolve_synth_e4 en compute_fid_kid.py).

No genera imagenes nuevas ni aplica IA generativa: solo compone una figura a
partir de archivos ya existentes. Las imagenes se ajustan (sin deformar ni
recortar) dentro de una celda cuadrada mediante padding blanco.
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
DEFAULT_SYNTH_CSV = Path("reports/results/synthetic_final_traceability.csv")
DEFAULT_SYNTH_IMAGE_DIR = Path("data/gen/SD_E2_HOPE/v1_8000")
DEFAULT_OUTPUT = Path("reports/figures/final/fig_real_vs_synthetic_grid.png")
DEFAULT_MANIFEST = Path("reports/results/real_vs_synthetic_grid_manifest.csv")
DEFAULT_SEED = 27
DEFAULT_NUM_PER_GROUP = 6
EXPECTED_REAL_TRAIN_MELANOMA = 1293
EXPECTED_SYNTH_E4 = 1293

REAL_GROUP = "real_train_melanoma"
SYNTH_GROUP = "synthetic_e4"
ROW_LABELS = {REAL_GROUP: "Real", SYNTH_GROUP: "Sintético"}
MANIFEST_COLUMNS = ["group", "position", "filename", "source_path", "selection_mode", "seed"]

# Rutas reservadas: si una ruta real resuelta cayera aqui, indicaria
# contaminacion con el pool mezclado de imagefolder o con el arbol de
# generacion sintetica, y el grupo real dejaria de ser fiable.
FORBIDDEN_REAL_SUBTREES = (Path("data/imagefolder"), Path("data/gen"))

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


def build_index_by_name(directory: Path) -> dict:
    return {p.name: p for p in directory.iterdir() if p.is_file()}


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
        rel = p[idx + 1 :]  # keep leading "data/..."
    return repo_root / rel


def resolve_real_candidates(repo_root: Path, master_csv: Path):
    """Real melanoma pool: split == 'train' & superclass == 'melanoma', resolved via
    the CSV's own 'path' column (original acquisition path), NOT via the
    (contaminated) data/imagefolder/train directory."""
    df = pd.read_csv(master_csv, low_memory=False)
    sub = df[(df["split"] == "train") & (df["superclass"] == "melanoma")]

    non_melanoma_flag = sub[sub["is_melanoma"] != 1]
    if len(non_melanoma_flag) > 0:
        fail(
            f"[real_train_melanoma] {len(non_melanoma_flag)} registros con "
            f"superclass=='melanoma' pero is_melanoma != 1; abortando por seguridad."
        )

    available = sub["image_name"].tolist()
    resolved = {}
    missing = []
    contaminated = []
    for name, raw_path in zip(sub["image_name"], sub["path"]):
        local = resolve_original_path(repo_root, raw_path)
        try:
            rel_to_root = local.resolve().relative_to(repo_root.resolve())
        except ValueError:
            rel_to_root = None
        if rel_to_root is not None and any(
            rel_to_root == sub_tree or sub_tree in rel_to_root.parents for sub_tree in FORBIDDEN_REAL_SUBTREES
        ):
            contaminated.append(name)
            continue
        if local.exists():
            resolved[name] = local
        else:
            missing.append(name)

    if contaminated:
        fail(
            f"[real_train_melanoma] {len(contaminated)} registros resuelven dentro de "
            f"arboles reservados (data/imagefolder o data/gen); el grupo real no puede "
            f"depender de esas rutas. Ejemplos: {contaminated[:5]}"
        )

    return available, resolved, missing


def resolve_synth_candidates(repo_root: Path, synth_csv: Path, synth_dir: Path):
    """Same filter + resolution logic as resolve_synth_e4() in compute_fid_kid.py."""
    df = pd.read_csv(synth_csv, low_memory=False)
    sub = df[df["used_in_E4_SYNTH"] == 1]
    available = sub["image_name"].tolist()
    index = build_index_by_name(synth_dir)
    resolved = {name: index[name] for name in available if name in index}
    missing = [name for name in available if name not in index]
    return available, resolved, missing


def deterministic_sample(resolved: dict, num: int, seed: int) -> list:
    """Sort resolvable keys lexicographically, then sample without replacement with an
    independent random.Random(seed). Reproducible regardless of dict/filesystem order."""
    keys = sorted(resolved.keys())
    if num > len(keys):
        fail(f"Se solicitaron {num} imagenes pero solo hay {len(keys)} resolubles.")
    return random.Random(seed).sample(keys, num)


def load_manual_filenames(filenames_file: Path) -> list:
    if not filenames_file.exists():
        fail(f"No se encontro el fichero de seleccion manual: {filenames_file}")

    if filenames_file.suffix.lower() == ".csv":
        df = pd.read_csv(filenames_file)
        column = next((c for c in df.columns if c.strip().lower() == "filename"), df.columns[0])
        values = [str(v).strip() for v in df[column].tolist()]
    else:
        values = [line.strip() for line in filenames_file.read_text(encoding="utf-8").splitlines()]

    return [v for v in values if v]


def select_manual(filenames_file: Path, resolved: dict, num: int, group_label: str, aliases: dict | None = None) -> list:
    """Resolve a manual filenames file against `resolved` keys. `aliases` optionally
    maps an alternate identifier (e.g. a real image's on-disk basename with
    extension) to the canonical key used in `resolved` (e.g. the CSV's
    extension-less image_name), so manual files can reference either form."""
    aliases = aliases or {}
    raw_selected = load_manual_filenames(filenames_file)

    if len(raw_selected) != num:
        fail(
            f"[{group_label}] '{filenames_file}' contiene {len(raw_selected)} filenames "
            f"pero --num-per-group es {num}."
        )

    dup_mask = pd.Series(raw_selected).duplicated(keep=False)
    if dup_mask.any():
        dups = sorted(set(pd.Series(raw_selected)[dup_mask]))
        fail(f"[{group_label}] filenames duplicados en seleccion manual: {dups}")

    canonical = []
    not_valid = []
    for value in raw_selected:
        if value in resolved:
            canonical.append(value)
        elif value in aliases and aliases[value] in resolved:
            canonical.append(aliases[value])
        else:
            not_valid.append(value)

    if not_valid:
        fail(
            f"[{group_label}] los siguientes filenames no pertenecen al conjunto valido "
            f"o no son resolubles fisicamente: {not_valid}"
        )

    return canonical


def select_group(
    group_label: str,
    resolved: dict,
    num: int,
    seed: int,
    manual_file: Path | None,
    aliases: dict | None = None,
):
    if manual_file is not None:
        selected = select_manual(manual_file, resolved, num, group_label, aliases=aliases)
        mode = "manual"
    else:
        selected = deterministic_sample(resolved, num, seed)
        repeat = deterministic_sample(resolved, num, seed)
        if selected != repeat:
            fail(f"[{group_label}] la seleccion aleatoria no es reproducible para seed={seed}.")
        mode = "random"
    return selected, mode


def pad_to_square(path: Path, cell_px: int) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return ImageOps.pad(im, (cell_px, cell_px), method=Image.LANCZOS, color=(255, 255, 255), centering=(0.5, 0.5))


def build_figure(real_images: list, synth_images: list, font_name: str) -> plt.Figure:
    cols = len(real_images)
    width_in = FIGURE_WIDTH_CM / 2.54
    height_in = width_in * (2 / cols) * 1.08  # margen extra vertical para los rotulos

    fig, axes = plt.subplots(2, cols, figsize=(width_in, height_in), squeeze=False)
    fig.patch.set_facecolor("white")

    for row_idx, (images, group) in enumerate([(real_images, REAL_GROUP), (synth_images, SYNTH_GROUP)]):
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


def write_manifest(manifest_path: Path, real_selected, real_resolved, real_mode, synth_selected, synth_resolved, synth_mode, seed: int) -> None:
    records = []
    for idx, key in enumerate(real_selected, start=1):
        records.append({
            "group": REAL_GROUP,
            "position": idx,
            "filename": real_resolved[key].name,
            "source_path": str(real_resolved[key]),
            "selection_mode": real_mode,
            "seed": seed,
        })
    offset = len(real_selected)
    for idx, key in enumerate(synth_selected, start=1):
        records.append({
            "group": SYNTH_GROUP,
            "position": offset + idx,
            "filename": synth_resolved[key].name,
            "source_path": str(synth_resolved[key]),
            "selection_mode": synth_mode,
            "seed": seed,
        })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    print(f"[OUTPUT] Manifest guardado en: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=None, help="Repo root (auto-detectado si se omite).")
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER_CSV)
    parser.add_argument("--synth-csv", type=Path, default=DEFAULT_SYNTH_CSV)
    parser.add_argument("--synth-image-dir", type=Path, default=DEFAULT_SYNTH_IMAGE_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla reproducible. Por defecto: 27.")
    parser.add_argument("--num-per-group", type=int, default=DEFAULT_NUM_PER_GROUP, help="Imagenes por grupo. Por defecto: 6.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--real-filenames-file",
        type=Path,
        default=None,
        help=(
            "TXT/CSV con seleccion manual de filenames reales (train melanoma). "
            "Acepta tanto el image_name del CSV maestro (sin extension) como el "
            "nombre de archivo real en disco (con extension)."
        ),
    )
    parser.add_argument(
        "--synthetic-filenames-file",
        type=Path,
        default=None,
        help="TXT/CSV con seleccion manual de filenames sinteticos (E4_SYNTH).",
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, p: Path) -> Path:
    return p if p.is_absolute() else repo_root / p


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path(__file__).resolve())

    master_csv = resolve_path(repo_root, args.master_csv)
    synth_csv = resolve_path(repo_root, args.synth_csv)
    synth_image_dir = resolve_path(repo_root, args.synth_image_dir)
    output_path = resolve_path(repo_root, args.output)
    manifest_path = resolve_path(repo_root, args.manifest)

    if not master_csv.exists():
        fail(f"No se encontro el CSV maestro: {master_csv}")
    if not synth_csv.exists():
        fail(f"No se encontro el CSV de trazabilidad sintetica: {synth_csv}")
    if not synth_image_dir.is_dir():
        fail(f"No se encontro el directorio de imagenes sinteticas: {synth_image_dir}")

    print("[AUDIT] === Grupo real (train melanoma, resuelto desde rutas originales del CSV maestro) ===")
    real_available, real_resolved, real_missing = resolve_real_candidates(repo_root, master_csv)
    print(f"[AUDIT] Fuente: {master_csv}")
    print("[AUDIT] Filtro: split==train, superclass==melanoma")
    print(f"[AUDIT] Disponibles: {len(real_available)}  Resolubles: {len(real_resolved)}  Faltantes: {len(real_missing)}")
    print("[AUDIT] Uses current data/imagefolder/train directly: NO (resuelto via columna 'path' del CSV maestro -> data/ogs/...)")
    if real_missing:
        print(f"[AUDIT][WARN] Ejemplos no resolubles: {real_missing[:5]}")
    if len(set(real_available)) != len(real_available):
        fail("[real_train_melanoma] se han detectado image_name duplicados en el CSV maestro filtrado.")
    if len(real_available) != EXPECTED_REAL_TRAIN_MELANOMA:
        print(
            f"[AUDIT][WARN] Se esperaban {EXPECTED_REAL_TRAIN_MELANOMA} melanomas reales de train "
            f"pero el filtro produce {len(real_available)}.",
            file=sys.stderr,
        )

    print("[AUDIT] === Grupo sintetico (E4_SYNTH) ===")
    synth_available, synth_resolved, synth_missing = resolve_synth_candidates(repo_root, synth_csv, synth_image_dir)
    print(f"[AUDIT] Fuente: {synth_csv}")
    print("[AUDIT] Filtro: used_in_E4_SYNTH == 1")
    print(f"[AUDIT] Disponibles: {len(synth_available)}  Resolubles: {len(synth_resolved)}  Faltantes: {len(synth_missing)}")
    if synth_missing:
        print(f"[AUDIT][WARN] Ejemplos no resolubles: {synth_missing[:5]}")
    if len(set(synth_available)) != len(synth_available):
        fail("[synthetic_e4] se han detectado image_name duplicados en el CSV de trazabilidad.")
    if len(synth_available) != EXPECTED_SYNTH_E4:
        print(
            f"[AUDIT][WARN] Se esperaban {EXPECTED_SYNTH_E4} imagenes E4_SYNTH pero el filtro "
            f"produce {len(synth_available)}.",
            file=sys.stderr,
        )

    real_aliases = {path.name: key for key, path in real_resolved.items()}
    real_selected, real_mode = select_group(
        REAL_GROUP, real_resolved, args.num_per_group, args.seed, args.real_filenames_file, aliases=real_aliases
    )
    synth_selected, synth_mode = select_group(
        SYNTH_GROUP, synth_resolved, args.num_per_group, args.seed, args.synthetic_filenames_file
    )

    if len(set(real_selected)) != len(real_selected):
        fail("[real_train_melanoma] duplicados en la seleccion final.")
    if len(set(synth_selected)) != len(synth_selected):
        fail("[synthetic_e4] duplicados en la seleccion final.")
    for key in real_selected:
        if not real_resolved[key].exists():
            fail(f"[real_train_melanoma] el archivo seleccionado no existe fisicamente: {real_resolved[key]}")
        if key not in set(real_available):
            fail(f"[real_train_melanoma] '{key}' no pertenece al conjunto split==train & superclass==melanoma.")
    for key in synth_selected:
        if not synth_resolved[key].exists():
            fail(f"[synthetic_e4] el archivo seleccionado no existe fisicamente: {synth_resolved[key]}")
        if key not in set(synth_available):
            fail(f"[synthetic_e4] '{key}' no pertenece al conjunto used_in_E4_SYNTH == 1.")

    print(f"[AUDIT] Seleccion real ({real_mode}, seed={args.seed}):")
    for idx, key in enumerate(real_selected, start=1):
        print(f"  {idx}  {real_resolved[key].name}")
    print(f"[AUDIT] Seleccion sintetica ({synth_mode}, seed={args.seed}):")
    for idx, key in enumerate(synth_selected, start=1):
        print(f"  {idx}  {synth_resolved[key].name}")

    font_name = register_calibri()

    real_images = [pad_to_square(real_resolved[k], CELL_PX) for k in real_selected]
    synth_images = [pad_to_square(synth_resolved[k], CELL_PX) for k in synth_selected]

    fig = build_figure(real_images, synth_images, font_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[OUTPUT] Figura guardada en: {output_path}")

    write_manifest(manifest_path, real_selected, real_resolved, real_mode, synth_selected, synth_resolved, synth_mode, args.seed)


if __name__ == "__main__":
    main()
