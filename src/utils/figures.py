#!/usr/bin/env python3
# Figures for the memory: before vs after filtering (blur and duplicates)

# Input and output paths
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

before_csv = Path("../../data/integrated/master_metadata_filtered1_enriched.csv")
after_csv  = Path("../../data/integrated/master_metadata_filtered2.csv")
out_dir    = Path("./data/reports")
out_dir.mkdir(parents=True, exist_ok=True)

# Parameters
BLUR_THR = 130.0   # Variance of Laplacian threshold used in filtering
FIG_DPI  = 180

# Read metadata
df_before = pd.read_csv(before_csv, low_memory=False)
df_after  = pd.read_csv(after_csv,  low_memory=False)

# Basic prep
df_before["superclass"] = df_before["superclass"].fillna("unknown")
df_after["superclass"]  = df_after["superclass"].fillna("unknown")

# Identify removed images
ids_before = set(df_before["image_name"].astype(str))
ids_after  = set(df_after["image_name"].astype(str))
removed_ids = ids_before - ids_after
df_removed = df_before[df_before["image_name"].astype(str).isin(removed_ids)].copy()

# 2) Blur histogram (before) with threshold
plt.figure(figsize=(6,4))
# Limitar el rango máximo a 50k para que no se aplane la distribución
plt.hist(df_before["blur"].astype(float), bins=60, range=(0, 10000))
plt.title("Blurriness distribution (before)")
plt.xlabel("Variance of Laplacian")
plt.ylabel("Images")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "02_blur_hist_before.png", dpi=FIG_DPI)
plt.close()

# 3) Blur histogram (removed)
if len(df_removed) > 0:
    plt.figure(figsize=(6,4))
    plt.hist(df_before["blur"].astype(float), bins=60, range=(0, 10000))
    plt.axvline(BLUR_THR, linestyle="--")
    plt.title("blur distribution (removed)")
    plt.xlabel("Variance of Laplacian")
    plt.ylabel("images")
    plt.tight_layout()
    plt.savefig(out_dir / "03_blur_hist_removed.png", dpi=FIG_DPI)
    plt.close()

# 4) Blur histogram (after)
plt.figure(figsize=(6,4))
plt.hist(df_before["blur"].astype(float), bins=60, range=(0, 10000))
plt.axvline(BLUR_THR, linestyle="--")
plt.title("blur distribution (after)")
plt.xlabel("Variance of Laplacian")
plt.ylabel("images")
plt.tight_layout()
plt.savefig(out_dir / "04_blur_hist_after.png", dpi=FIG_DPI)
plt.close()