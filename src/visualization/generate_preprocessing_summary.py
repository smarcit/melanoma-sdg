#!/usr/bin/env python3
"""Figura cuantitativa resumen del pipeline de preprocesamiento.

Pipeline real reconstruido:
    Datos integrados -> control de integridad -> enriquecimiento
    -> filtrado por blur -> eliminacion de duplicados exactos (MD5)
    -> eliminacion de casi duplicados (pHash/SSIM) -> dataset final

El enriquecimiento no elimina filas y la asignacion de splits no es un
filtro: ninguno de los dos aparece en esta figura.

Fuente de valores: reports/results/preprocessing_filter_summary.csv
(generado por auditoria previa, ya validada). Este script no vuelve a
ejecutar el pipeline de preprocesamiento ni modifica ningun CSV/dataset
existente; unicamente lee el resumen y dibuja la figura, validando antes
las identidades aritmeticas.

Genera unicamente:
    reports/figures/final/fig_preprocessing_quantitative_summary.png
    reports/figures/final/fig_preprocessing_quantitative_summary.pdf
"""

from __future__ import annotations

import argparse
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

DEFAULT_CSV = Path("reports/results/preprocessing_filter_summary.csv")
DEFAULT_OUTPUT_PNG = Path("reports/figures/final/fig_preprocessing_quantitative_summary.png")
DEFAULT_OUTPUT_PDF = Path("reports/figures/final/fig_preprocessing_quantitative_summary.pdf")

DPI = 300
FIGURE_WIDTH_IN = 16.0 / 2.54
FIGURE_HEIGHT_IN = 8.6 / 2.54

COLOR_RETAINED = "#1174E6"
COLOR_REMOVED = "#ED7D31"
COLOR_TEXT = "#3F4A5A"
COLOR_GRID = "#D9D9D9"

STAGE_ORDER = ["integrated", "integrity", "blur", "exact_duplicates", "near_duplicates"]

# Valores auditados y validados (hardcoded como red de seguridad ante ediciones
# accidentales del CSV): el script falla si el CSV no coincide exactamente.
EXPECTED = {
    "integrated": {"before": 64172, "removed": 0, "after": 64172},
    "integrity": {"before": 64172, "removed": 2020, "after": 62152},
    "blur": {"before": 62152, "removed": 3108, "after": 59044},
    "exact_duplicates": {"before": 59044, "removed": 10029, "after": 49015},
    "near_duplicates": {"before": 49015, "removed": 730, "after": 48285},
}

PANEL_A_LABELS = {
    "integrated": "Datos integrados",
    "integrity": "Tras integridad",
    "blur": "Tras desenfoque",
    "exact_duplicates": "Tras duplicados exactos",
    "near_duplicates": "Dataset final",
}

PANEL_B_LABELS = {
    "integrity": "Integridad",
    "blur": "Desenfoque",
    "exact_duplicates": "Duplicados exactos (MD5)",
    "near_duplicates": "Casi duplicados (pHash/SSIM)",
}

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


def es_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def es_pct(value: float, decimals: int) -> str:
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}".replace(".", ",")


def load_summary(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        fail(f"No se encontro el CSV resumen: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"stage", "before", "removed", "after", "removed_from_previous_pct", "retained_from_start_pct"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        fail(f"Faltan columnas en '{csv_path}': {sorted(missing_cols)}")

    df = df.set_index("stage")
    missing_stages = [s for s in STAGE_ORDER if s not in df.index]
    if missing_stages:
        fail(f"Faltan etapas en '{csv_path}': {missing_stages}")

    print(f"[AUDIT] Fuente: {csv_path}")
    return df


def validate_summary(df: pd.DataFrame) -> None:
    initial_total = int(df.loc["integrated", "before"])
    final_total = int(df.loc["near_duplicates", "after"])

    if initial_total != 64172:
        fail(f"El total inicial esperado es 64172, pero el CSV indica {initial_total}.")
    if final_total != 48285:
        fail(f"El total final esperado es 48285, pero el CSV indica {final_total}.")

    prev_after = initial_total
    for stage in STAGE_ORDER:
        expected = EXPECTED[stage]
        row = df.loc[stage]
        before, removed, after = int(row["before"]), int(row["removed"]), int(row["after"])

        if (before, removed, after) != (expected["before"], expected["removed"], expected["after"]):
            fail(
                f"Etapa '{stage}': valores del CSV {before, removed, after} no coinciden "
                f"con los valores auditados {expected['before'], expected['removed'], expected['after']}."
            )
        if before - removed != after:
            fail(f"Etapa '{stage}': la identidad before - removed = after no se cumple ({before} - {removed} != {after}).")
        if stage != "integrated" and before != prev_after:
            fail(f"Etapa '{stage}': 'before' ({before}) no coincide con el 'after' de la etapa previa ({prev_after}).")
        prev_after = after

    print("[AUDIT] Identidades validadas:")
    print(f"[AUDIT]   64172 - 2020  = 62152 -> {64172 - 2020 == 62152}")
    print(f"[AUDIT]   62152 - 3108  = 59044 -> {62152 - 3108 == 59044}")
    print(f"[AUDIT]   59044 - 10029 = 49015 -> {59044 - 10029 == 49015}")
    print(f"[AUDIT]   49015 - 730   = 48285 -> {49015 - 730 == 48285}")
    overall_retention = 100.0 * final_total / initial_total
    print(f"[AUDIT] Dataset inicial: {initial_total} | Dataset final: {final_total}")
    print(f"[AUDIT] Retencion global: {overall_retention:.2f} %")
    print("[AUDIT] Enrichment no se trata como perdida de filas (no aparece en el CSV/figura).")
    print("[AUDIT] Split assignment no se trata como filtro (no aparece en el CSV/figura).")
    print("[AUDIT] No se utilizan datos sinteticos en esta figura.")


def build_figure(df: pd.DataFrame, font_name: str, output_png: Path, output_pdf: Path | None) -> None:
    plt.rcParams["font.family"] = font_name
    plt.rcParams["font.size"] = 8.5
    plt.rcParams["text.color"] = COLOR_TEXT
    plt.rcParams["axes.edgecolor"] = "#808080"
    plt.rcParams["axes.labelcolor"] = COLOR_TEXT
    plt.rcParams["xtick.color"] = COLOR_TEXT
    plt.rcParams["ytick.color"] = COLOR_TEXT

    initial_total = int(df.loc["integrated", "before"])

    panel_a_stages = STAGE_ORDER
    panel_a_labels = [PANEL_A_LABELS[s] for s in panel_a_stages]
    panel_a_values = [int(df.loc[s, "after"]) for s in panel_a_stages]
    panel_a_pcts = [float(df.loc[s, "retained_from_start_pct"]) for s in panel_a_stages]

    panel_b_stages = [s for s in STAGE_ORDER if s != "integrated"]
    panel_b_labels = [PANEL_B_LABELS[s] for s in panel_b_stages]
    panel_b_values = [int(df.loc[s, "removed"]) for s in panel_b_stages]
    panel_b_pcts = [float(df.loc[s, "removed_from_previous_pct"]) for s in panel_b_stages]

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=DPI)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.55, 0.45], wspace=1.9, left=0.17, right=0.97, top=0.88, bottom=0.14)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    bar_height = 0.55

    # Panel A - retenidas
    y_a = list(range(len(panel_a_labels)))
    ax_a.set_facecolor("white")
    ax_a.grid(axis="x", color=COLOR_GRID, linewidth=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.barh(y_a, panel_a_values, height=bar_height, color=COLOR_RETAINED, zorder=3, edgecolor="none")
    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels(panel_a_labels, fontsize=8.5)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Numero de imagenes", fontsize=8.5)
    ax_a.set_title("Imagenes retenidas", fontsize=10.5, fontweight="bold", color="#222222", pad=8)
    ax_a.set_xlim(0, initial_total * 1.30)
    ax_a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: es_int(int(x))))
    for spine in ("top", "right", "left"):
        ax_a.spines[spine].set_visible(False)
    ax_a.spines["bottom"].set_color("#808080")
    ax_a.tick_params(axis="y", length=0)

    for y, value, pct in zip(y_a, panel_a_values, panel_a_pcts):
        label = f"{es_int(value)} ({es_pct(pct, 1)} %)"
        ax_a.text(value + initial_total * 0.02, y, label, va="center", ha="left", fontsize=8, color=COLOR_TEXT)

    # Panel B - eliminadas
    y_b = list(range(len(panel_b_labels)))
    max_removed = max(panel_b_values)
    ax_b.set_facecolor("white")
    ax_b.grid(axis="x", color=COLOR_GRID, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.barh(y_b, panel_b_values, height=bar_height, color=COLOR_REMOVED, zorder=3, edgecolor="none")
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(panel_b_labels, fontsize=8.5)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Numero de imagenes", fontsize=8.5)
    ax_b.set_title("Imagenes eliminadas por criterio", fontsize=10.5, fontweight="bold", color="#222222", pad=8)
    ax_b.set_xlim(0, max_removed * 1.40)
    ax_b.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: es_int(int(x))))
    for spine in ("top", "right", "left"):
        ax_b.spines[spine].set_visible(False)
    ax_b.spines["bottom"].set_color("#808080")
    ax_b.tick_params(axis="y", length=0)

    for y, value, pct in zip(y_b, panel_b_values, panel_b_pcts):
        label = f"{es_int(value)} ({es_pct(pct, 2)} %)"
        ax_b.text(value + max_removed * 0.02, y, label, va="center", ha="left", fontsize=8, color=COLOR_TEXT)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    print(f"[AUDIT] Figura PNG guardada en: {output_png}")

    if output_pdf is not None:
        fig.savefig(output_pdf, facecolor="white", bbox_inches="tight", pad_inches=0.08)
        print(f"[AUDIT] Figura PDF guardada en: {output_pdf}")

    plt.close(fig)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None, help="Raiz del repositorio (auto-detectada si se omite).")
    parser.add_argument("--csv", type=Path, default=None, help="Ruta al CSV resumen (por defecto: reports/results/preprocessing_filter_summary.csv).")
    parser.add_argument("--output-png", type=Path, default=None, help="Ruta de salida PNG.")
    parser.add_argument("--output-pdf", type=Path, default=None, help="Ruta de salida PDF (usa 'none' para omitir).")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root or find_repo_root(Path(__file__).resolve())

    csv_path = args.csv if args.csv is not None else repo_root / DEFAULT_CSV
    output_png = args.output_png if args.output_png is not None else repo_root / DEFAULT_OUTPUT_PNG
    if args.output_pdf is not None and str(args.output_pdf).lower() == "none":
        output_pdf = None
    else:
        output_pdf = args.output_pdf if args.output_pdf is not None else repo_root / DEFAULT_OUTPUT_PDF

    df = load_summary(csv_path)
    validate_summary(df)

    font_name = register_calibri()
    build_figure(df, font_name, output_png, output_pdf)

    print("[AUDIT] Generacion completada correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
