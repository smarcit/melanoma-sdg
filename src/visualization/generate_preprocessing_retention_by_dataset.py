#!/usr/bin/env python3
"""Figura complementaria: impacto global del preprocesamiento por dataset real.

Para cada dataset real se compara el conteo inicial (tras integracion) con el
conteo final (tras integridad + blur + duplicados exactos + casi duplicados,
de forma secuencial). La asignacion de splits NO se trata como filtrado y no
aparece en esta figura.

Fuente de valores: conteos auditados (hardcoded abajo, como en
generate_preprocessing_summary.py) documentados en la auditoria de
preprocesamiento. Los porcentajes NO se hardcodean: se calculan en este
script a partir de initial_count/final_count y luego se contrastan contra los
porcentajes esperados de la auditoria como red de seguridad.

Genera unicamente:
    reports/results/preprocessing_retention_by_dataset.csv
    reports/figures/final/fig_preprocessing_retention_by_dataset.png
    reports/figures/final/fig_preprocessing_retention_by_dataset.pdf
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

DEFAULT_OUTPUT_CSV = Path("reports/results/preprocessing_retention_by_dataset.csv")
DEFAULT_OUTPUT_PNG = Path("reports/figures/final/fig_preprocessing_retention_by_dataset.png")
DEFAULT_OUTPUT_PDF = Path("reports/figures/final/fig_preprocessing_retention_by_dataset.pdf")

DPI = 300
FIGURE_WIDTH_IN = 16.0 / 2.54
FIGURE_HEIGHT_IN = 9.0 / 2.54

COLOR_RETAINED = "#1174E6"
COLOR_REMOVED = "#ED7D31"
COLOR_TEXT = "#3F4A5A"
COLOR_GRID = "#D9D9D9"

# Conteos auditados (initial = tras integracion de datasets reales; final = tras
# integridad + blur + duplicados exactos + casi duplicados, secuencialmente).
# No incluyen asignacion de splits.
AUDITED_COUNTS = {
    "ISIC2019": {"initial": 25331, "final": 12828},
    "HAM10000": {"initial": 10015, "final": 9802},
    "DERM12345": {"initial": 12345, "final": 11290},
    "PH2": {"initial": 200, "final": 200},
    "Fitzpatrick17k": {"initial": 16281, "final": 14165},
}

# Porcentajes esperados de la auditoria (SOLO como valores de validacion; el
# script calcula los porcentajes reales a partir de AUDITED_COUNTS y falla si
# se desvian de estos mas alla de la tolerancia).
EXPECTED_RETAINED_PCT = {
    "ISIC2019": 50.64,
    "HAM10000": 97.87,
    "DERM12345": 91.45,
    "PH2": 100.00,
    "Fitzpatrick17k": 87.00,
}
VALIDATION_TOLERANCE_PCT = 0.05

# Orden solicitado, de arriba hacia abajo en la figura.
DATASET_ORDER = ["HAM10000", "DERM12345", "ISIC2019", "PH2", "Fitzpatrick17k"]

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


def es_pct(value: float, decimals: int) -> str:
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}".replace(".", ",")


def build_dataframe() -> pd.DataFrame:
    rows = []
    for dataset, counts in AUDITED_COUNTS.items():
        initial = int(counts["initial"])
        final = int(counts["final"])
        removed = initial - final
        retained_pct = 100.0 * final / initial
        removed_pct = 100.0 * removed / initial
        rows.append(
            {
                "dataset": dataset,
                "initial_count": initial,
                "final_count": final,
                "removed_count": removed,
                "retained_pct": retained_pct,
                "removed_pct": removed_pct,
            }
        )
    df = pd.DataFrame(rows).set_index("dataset")
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    if len(df) != 5:
        fail(f"Se esperaban exactamente 5 datasets, se encontraron {len(df)}.")

    missing = set(DATASET_ORDER) - set(df.index)
    if missing:
        fail(f"Faltan datasets en los conteos auditados: {sorted(missing)}")

    for dataset, row in df.iterrows():
        initial = int(row["initial_count"])
        final = int(row["final_count"])
        removed = int(row["removed_count"])

        if initial - removed != final:
            fail(f"'{dataset}': initial_count - removed_count != final_count ({initial} - {removed} != {final}).")
        if final > initial:
            fail(f"'{dataset}': final_count ({final}) > initial_count ({initial}).")

        total_pct = row["retained_pct"] + row["removed_pct"]
        if abs(total_pct - 100.0) > 1e-6:
            fail(f"'{dataset}': retained_pct + removed_pct = {total_pct}, no suma 100 %.")

        expected = EXPECTED_RETAINED_PCT[dataset]
        if abs(row["retained_pct"] - expected) > VALIDATION_TOLERANCE_PCT:
            fail(
                f"'{dataset}': retained_pct calculado ({row['retained_pct']:.2f} %) se desvia del "
                f"esperado por auditoria ({expected:.2f} %) mas alla de la tolerancia ({VALIDATION_TOLERANCE_PCT} %)."
            )

    ph2_retained = df.loc["PH2", "retained_pct"]
    if abs(ph2_retained - 100.0) > 1e-6:
        fail(f"PH2 deberia retener el 100 % de las imagenes; se calculo {ph2_retained:.4f} %.")

    print("[AUDIT] Validaciones superadas:")
    print("[AUDIT]   Exactamente 5 datasets.")
    print("[AUDIT]   initial_count - removed_count == final_count para todos los datasets.")
    print("[AUDIT]   retained_pct + removed_pct == 100 % para todos los datasets.")
    print("[AUDIT]   Porcentajes calculados coinciden con los esperados de auditoria (tolerancia "
          f"{VALIDATION_TOLERANCE_PCT} %).")
    print("[AUDIT]   PH2 = 100 % retenido.")
    print("[AUDIT]   No se utilizan datos sinteticos en esta figura.")
    print("[AUDIT]   La asignacion de splits no se trata como filtrado (no aparece en la figura/CSV).")
    for dataset in DATASET_ORDER:
        row = df.loc[dataset]
        print(
            f"[AUDIT]   {dataset}: initial={int(row['initial_count'])} final={int(row['final_count'])} "
            f"removed={int(row['removed_count'])} retained={row['retained_pct']:.2f}% removed={row['removed_pct']:.2f}%"
        )


def write_csv(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ordered = df.loc[DATASET_ORDER].reset_index()
    ordered = ordered[["dataset", "initial_count", "final_count", "removed_count", "retained_pct", "removed_pct"]]
    ordered.to_csv(output_csv, index=False)
    print(f"[AUDIT] CSV de trazabilidad guardado en: {output_csv}")


def build_figure(df: pd.DataFrame, font_name: str, output_png: Path, output_pdf: Path | None) -> None:
    plt.rcParams["font.family"] = font_name
    plt.rcParams["font.size"] = 9.5
    plt.rcParams["text.color"] = COLOR_TEXT
    plt.rcParams["axes.edgecolor"] = "#808080"
    plt.rcParams["axes.labelcolor"] = COLOR_TEXT
    plt.rcParams["xtick.color"] = COLOR_TEXT
    plt.rcParams["ytick.color"] = COLOR_TEXT

    # Orden de la figura: de arriba hacia abajo segun DATASET_ORDER.
    # matplotlib.barh dibuja de abajo hacia arriba, asi que se invierte el eje Y.
    labels = DATASET_ORDER
    retained_pcts = [float(df.loc[d, "retained_pct"]) for d in labels]
    removed_pcts = [float(df.loc[d, "removed_pct"]) for d in labels]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.20, right=0.96, top=0.86, bottom=0.14)

    y_pos = list(range(len(labels)))
    bar_height = 0.55

    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    bars_retained = ax.barh(
        y_pos, retained_pcts, height=bar_height, color=COLOR_RETAINED,
        edgecolor="none", zorder=3, label="Retenido",
    )
    bars_removed = ax.barh(
        y_pos, removed_pcts, left=retained_pcts, height=bar_height, color=COLOR_REMOVED,
        edgecolor="none", zorder=3, label="Eliminado",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{int(round(x))}"))
    ax.set_xlabel("Porcentaje de imagenes (%)", fontsize=9.5)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#808080")
    ax.tick_params(axis="y", length=0)

    # Etiquetas de porcentaje: dentro del segmento si hay espacio suficiente,
    # fuera (a la derecha de la barra) si el segmento es demasiado pequeno para
    # que el texto quepa legible. PH2 (0,0 % eliminado) no muestra etiqueta de
    # eliminado para evitar ruido visual.
    MIN_INSIDE_PCT = 6.0

    for y, pct in zip(y_pos, retained_pcts):
        label = f"{es_pct(pct, 1)} %"
        if pct >= MIN_INSIDE_PCT:
            ax.text(pct / 2.0, y, label, va="center", ha="center", fontsize=8.5, color="white")
        else:
            ax.text(pct + 1.2, y, label, va="center", ha="left", fontsize=8, color=COLOR_TEXT)

    for y, retained, removed in zip(y_pos, retained_pcts, removed_pcts):
        if removed <= 0.0:
            continue
        label = f"{es_pct(removed, 1)} %"
        if removed >= MIN_INSIDE_PCT:
            ax.text(retained + removed / 2.0, y, label, va="center", ha="center", fontsize=8.5, color="white")
        else:
            ax.text(min(retained + removed + 1.2, 99.0), y, label, va="center", ha="left", fontsize=8, color=COLOR_TEXT)

    ax.legend(
        handles=[bars_retained, bars_removed],
        labels=["Retenido", "Eliminado"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handlelength=1.2,
        handleheight=1.0,
        columnspacing=1.4,
    )

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
    parser.add_argument("--output-csv", type=Path, default=None, help="Ruta de salida del CSV de trazabilidad.")
    parser.add_argument("--output-png", type=Path, default=None, help="Ruta de salida PNG.")
    parser.add_argument("--output-pdf", type=Path, default=None, help="Ruta de salida PDF (usa 'none' para omitir).")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root or find_repo_root(Path(__file__).resolve())

    output_csv = args.output_csv if args.output_csv is not None else repo_root / DEFAULT_OUTPUT_CSV
    output_png = args.output_png if args.output_png is not None else repo_root / DEFAULT_OUTPUT_PNG
    if args.output_pdf is not None and str(args.output_pdf).lower() == "none":
        output_pdf = None
    else:
        output_pdf = args.output_pdf if args.output_pdf is not None else repo_root / DEFAULT_OUTPUT_PDF

    df = build_dataframe()
    validate_dataframe(df)
    write_csv(df, output_csv)

    font_name = register_calibri()
    build_figure(df, font_name, output_png, output_pdf)

    print("[AUDIT] Generacion completada correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
