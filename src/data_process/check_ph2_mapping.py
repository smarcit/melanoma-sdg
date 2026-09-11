#!/usr/bin/env python3
import pandas as pd

MM = "../../data/master_metadata_v3.csv"          # ajusta si usas otro nombre
TM = "../../data/taxonomy_map.csv"

df = pd.read_csv(MM)
tm = pd.read_csv(TM)

# etiquetas PH2 presentes en master
ph2 = df[df["dataset"]=="PH2"]
labels = ph2["original_label"].fillna("").str.strip().unique().tolist()

# etiquetas PH2 presentes en taxonomy_map
tm_ph2 = tm[tm["dataset"].str.strip().str.lower()=="ph2"].copy()
tm_ph2["original_label"] = tm_ph2["original_label"].fillna("").str.strip()

in_tm = set(tm_ph2["original_label"].tolist())
missing = sorted([l for l in labels if l and l not in in_tm])

print("\nEtiquetas PH2 detectadas en master_metadata:")
for l in sorted([x for x in labels if x]): print("  -", l)

print("\nEtiquetas PH2 faltantes en taxonomy_map:")
if missing:
    for l in missing: print("  -", l)
else:
    print("  (ninguna)")

# Extra: recuento por superclass en PH2 tras el cruce (para ver que se llenó)
print("\nDistribución PH2 por superclass (después del mapeo):")
print(ph2["superclass"].value_counts(dropna=False))
