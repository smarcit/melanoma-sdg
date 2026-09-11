#!/usr/bin/env python3
# Input and output metadata paths
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import hashlib
import imagehash

in_csv = Path("../../data/synth_metadata_filtered1.csv")
out_csv = Path("../../data/synth_metadata_filtered1_enriched.csv")

# Read metadata
df = pd.read_csv(in_csv, low_memory=False)

# Compute blur (Variance of Laplacian)
blur_vals = []
for fp in df["path"]:
    with Image.open(fp).convert("L") as im:
        arr = np.array(im, dtype=np.uint8)
    lap = cv2.Laplacian(arr, cv2.CV_64F, ksize=3)
    blur_vals.append(lap.var())
df["blur"] = blur_vals

# Compute md5
md5_vals = []
for fp in df["path"]:
    with open(fp, "rb") as f:
        md5_vals.append(hashlib.md5(f.read()).hexdigest())
df["md5"] = md5_vals

# Compute perceptual hash (store as unsigned integer)
phash_vals = []
for fp in df["path"]:
    with Image.open(fp).convert("RGB") as im:
        h = imagehash.phash(im)          # ImageHash object
        phash_vals.append(int(str(h), 16))  # convert from hex string to int
df["phash"] = phash_vals  # 64-bit integer

# Save enriched metadata
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

