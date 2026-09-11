#!/usr/bin/env python3
# Input and output paths
from pathlib import Path
import pandas as pd

in_csv = Path("../../data/integrated/master_metadata_split.csv")
out_dir = Path("../../data/integrated")
out_dir.mkdir(parents=True, exist_ok=True)

# Read metadata
df = pd.read_csv(in_csv)

# Prepare helper columns
df["superclass_report"] = df["superclass"].fillna("unknown")

# Dataset balance per split (counts and percentages within split)
ds_counts = (
    df.groupby(["split", "dataset"], as_index=False)
      .size()
      .rename(columns={"size": "count"})
)
totals_by_split = ds_counts.groupby("split", as_index=False)["count"].sum().rename(columns={"count":"total_split"})
ds_balance = ds_counts.merge(totals_by_split, on="split")
ds_balance["percent_in_split"] = (ds_balance["count"] / ds_balance["total_split"] * 100.0).round(2)

# Class balance per split (counts and percentages within split)
cl_counts = (
    df.groupby(["split", "superclass_report"], as_index=False)
      .size()
      .rename(columns={"size": "count"})
)
totals_by_split2 = cl_counts.groupby("split", as_index=False)["count"].sum().rename(columns={"count":"total_split"})
cl_balance = cl_counts.merge(totals_by_split2, on="split")
cl_balance["percent_in_split"] = (cl_balance["count"] / cl_balance["total_split"] * 100.0).round(2)

# Class-by-dataset table per split (pivot with counts)
pivot_table = (
    df.pivot_table(index=["split","dataset"], columns="superclass_report", values="image_name", aggfunc="count", fill_value=0)
      .reset_index()
)

# Write outputs
ds_balance.to_csv(out_dir / "dataset_balance_by_split.csv", index=False)
cl_balance.to_csv(out_dir / "class_balance_by_split.csv", index=False)
pivot_table.to_csv(out_dir / "split_dataset_class_pivot.csv", index=False)

