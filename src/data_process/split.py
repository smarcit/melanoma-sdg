#!/usr/bin/env python3
# Input and output metadata paths
from pathlib import Path
import pandas as pd
import numpy as np

in_csv = Path("../../data/integrated/master_metadata_filtered2.csv")
out_csv = Path("../../data/integrated/master_metadata_split.csv")

# Fixed seed for reproducibility
np.random.seed(666)

# Read metadata
df = pd.read_csv(in_csv)

# Initialize split column
df["split"] = ""

# Assign test splits
df.loc[df["dataset"] == "ISIC2019", "split"] = "test"
df.loc[df["dataset"] == "PH2", "split"] = "test_control"
df.loc[df["dataset"] == "Fitzpatrick17k", "split"] = "test_fairness"

# Helper columns for grouping
df["dup_group_id"] = df["dup_group_id"].fillna("")
df["group_unit"] = np.where(df["dup_group_id"] != "", df["dup_group_id"], df["image_name"])

# Train/val from HAM10000 and DERM12345 with class-stratified 10% per dataset
for ds in ["HAM10000", "DERM12345"]:
    sub = df[(df["dataset"] == ds) & (df["split"] == "")]
    # stratify by superclass at group level
    for sc in sub["superclass"].dropna().unique().tolist():
        sub_sc = sub[sub["superclass"] == sc]
        groups = sub_sc["group_unit"].drop_duplicates().to_numpy()
        if len(groups) == 0:
            continue
        idx = np.random.permutation(len(groups))
        n_val = int(np.floor(0.10 * len(groups)))
        if n_val == 0 and len(groups) > 0:
            n_val = 1
        val_groups = set(groups[idx[:n_val]])
        df.loc[(df["dataset"] == ds) & (df["superclass"] == sc) & (df["group_unit"].isin(val_groups)), "split"] = "val"
        df.loc[(df["dataset"] == ds) & (df["superclass"] == sc) & (~df["group_unit"].isin(val_groups)), "split"] = "train"
    # handle rows with missing superclass
    sub_nan = sub[sub["superclass"].isna()]
    if len(sub_nan) > 0:
        groups = sub_nan["group_unit"].drop_duplicates().to_numpy()
        idx = np.random.permutation(len(groups))
        n_val = int(np.floor(0.10 * len(groups)))
        if n_val == 0 and len(groups) > 0:
            n_val = 1
        val_groups = set(groups[idx[:n_val]])
        df.loc[(df["dataset"] == ds) & (df["superclass"].isna()) & (df["group_unit"].isin(val_groups)), "split"] = "val"
        df.loc[(df["dataset"] == ds) & (df["superclass"].isna()) & (~df["group_unit"].isin(val_groups)), "split"] = "train"

# Drop helper column and write output
df = df.drop(columns=["group_unit"])
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

