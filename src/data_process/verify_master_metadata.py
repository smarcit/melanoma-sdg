#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

def main():
    df = pd.read_csv("../../data/master_metadata_v3.csv")

    print("\n=== Columnas presentes ===")
    print(df.columns.tolist())

    print("\n=== Nº de filas ===")
    print(len(df))

    print("\n=== Duplicados image_id ===")
    dup = df["image_id"].duplicated().sum()
    print(f"{dup} duplicados")

    print("\n=== Distribución por dataset ===")
    print(df["dataset"].value_counts())

    print("\n=== Distribución por superclass ===")
    print(df["superclass"].value_counts(dropna=False))

    print("\n=== Distribución binary (0/1) ===")
    print(df["binary"].value_counts(dropna=False))

    print("\n=== Corruptos ===")
    print(df["is_corrupt"].value_counts())

    print("\n=== is_rgb vs channels ===")
    print(df.groupby(["is_rgb","channels"]).size())

    print("\n=== Valores ausentes por columna (top 15) ===")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    print("\n=== Ejemplo de filas con info extra (HAM10000) ===")
    print(df[df["dataset"]=="HAM10000"].head(5)[["age","sex","anatom_site_general","diagnosis_method"]])

    print("\n=== Ejemplo de filas PH2 ===")
    cols = ["clinical_diagnosis","dermoscopic_diagnosis","histological_diagnosis","fitzpatrick","asymmetry","border","color","diameter"]
    print(df[df["dataset"]=="PH2"].head(5)[cols])

    print("\n=== Ejemplo de filas DERM12345 ===")
    print(df[df["dataset"]=="DERM12345"].head(5)[["anatom_site_general","license"]])

    # Chequeo: dimensiones válidas
    bad_dims = df[(df["width"] <= 0) | (df["height"] <= 0)]
    print(f"\nImágenes con dimensiones inválidas: {len(bad_dims)}")

if __name__ == "__main__":
    main()
