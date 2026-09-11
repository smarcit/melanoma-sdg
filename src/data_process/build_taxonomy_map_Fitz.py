#!/usr/bin/env python3
# Input and output paths
from pathlib import Path
import pandas as pd

in_csv = Path("../../data/ogs/Fitzpatrick17k/fitzpatrick17k_metadata.csv")
out_csv = Path("../../data/taxonomy_map_fitzpatrick17k.csv")

# Read metadata
df = pd.read_csv(in_csv)

# Unique labels
labels = sorted(df["label"].dropna().astype(str).str.strip().unique().tolist())

# Simple keyword rules -> (superclass, binary)
# Notes:
# - malignant -> binary=1
# - known benign superclasses keep binary=0
# - everything else -> other, 0
rules = [
    # melanoma and variants
    (["melanoma", "lentigo maligna", "lmm"], ("melanoma", 1)),
    # bcc
    (["basal cell carcinoma", "bcc"], ("basal_cell_carcinoma", 1)),
    # scc and related
    (["squamous cell carcinoma", "scc", "keratoacanthoma"], ("squamous_cell_carcinoma", 1)),
    # actinic / bowen
    (["actinic keratosis", "bowen"], ("actinic_keratosis_bowen", 0)),
    # seborrheic keratosis
    (["seborrheic keratosis", "seborrhoeic keratosis"], ("benign_keratosis", 0)),
    # melanocytic nevus (naevus spellings included)
    (["nevus", "naevus", "blue nevus", "congenital nevus"], ("melanocytic_nevus", 0)),
    # dermatofibroma
    (["dermatofibroma"], ("dermatofibroma", 0)),
    # vascular lesions (common clinical terms)
    (["vascular lesion", "angioma", "hemangioma", "cherry angioma", "venous lake", "telangiectasia", "pyogenic granuloma"], ("vascular_lesion", 0)),
]

# Mapping function
def map_label_to_taxonomy(lbl: str):
    l = lbl.strip().lower()
    for keywords, (sc, binv) in rules:
        for kw in keywords:
            if kw in l:
                return sc, binv
    return "other", 0

# Build taxonomy map rows
rows = []
for lbl in labels:
    sc, binv = map_label_to_taxonomy(lbl)
    rows.append({
        "dataset": "Fitzpatrick17k",
        "original_label": lbl,
        "superclass": sc,
        "binary": binv
    })

# Write taxonomy map csv
out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=["dataset","original_label","superclass","binary"]).to_csv(out_csv, index=False)

