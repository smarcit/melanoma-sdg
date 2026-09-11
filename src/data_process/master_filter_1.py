#!/usr/bin/env python3

# Metadata input and output paths
from pathlib import Path
import pandas as pd

input = Path("../../data/synth_metadata.csv")
output = Path("../../data/integrated/synth_metadata_filtered1.csv")

# Minimum image size
min_side = 256

# Read metadata
df = pd.read_csv(input)

# Boolean masks
m_corrupt = df["is_corrupt"] == True
m_nonrgb = df["is_rgb"] == False
m_badch = df["channels"] != 3
m_badsize = df[["width", "height"]].min(axis=1) < min_side

# Filter unwanted rows
keep_mask = ~(m_corrupt | m_nonrgb | m_badch | m_badsize)
df_out = df.loc[keep_mask].copy()

# Write output
output.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(output, index=False)