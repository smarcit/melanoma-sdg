# Simple checker for taxonomy_map_fitzpatrick17k.csv
import pandas as pd, re

# Input path
csv_path = "../../data/taxonomy_map_fitzpatrick17k.csv"

# Allowed superclasses (alineado con ISIC + extras)
allowed_super = {
    "melanoma","melanocytic_nevus","basal_cell_carcinoma","squamous_cell_carcinoma",
    "actinic_keratosis_bowen","benign_keratosis","dermatofibroma","vascular_lesion",
    "other_benign","other_malignant"
}

# Keyword buckets (minúsculas)
MALIG = [
    r"melanom", r"lentigo\s*malign", r"\bllm\b", r"\bbcc\b|basal cell",
    r"\bscc\b|squamous cell", r"bowen", r"carcinoma in situ",
    r"keratoacanthoma", r"kaposi"
]
# Trátalos como malignos para comparabilidad con ISIC
PRECANCER_POS = [r"actinic keratosis", r"\bakiec\b"]

BENIGN = [
    r"nevus|naevus|mole|spitz|blue nevus|dysplastic", r"seborr(he)?ic keratosis|benign keratosis|lplk|solar lentigo|lentigo simplex",
    r"dermatofibroma", r"angioma|hemangioma|pyogenic granuloma",
    r"wart|verruca|mollusc", r"lipoma|keloid|scar|cyst",
    r"psoriasis|eczema|dermatitis|rosacea|urticaria|vitiligo|acne",
    r"tinea|impetigo|herpes zoster|shingles"
]
AMBIG = [r"cutaneous horn", r"lentigo(?!\s*malign)"]  # revisar manualmente

def any_match(text, patterns):
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

df = pd.read_csv(csv_path)

# Normaliza columnas esperadas: dataset, original_label, superclass, binary
for c in ["dataset","original_label","superclass","binary"]:
    assert c in df.columns, f"Missing column: {c}"

issues = []

for i, r in df.iterrows():
    lab = str(r["original_label"])
    sc  = str(r["superclass"]).strip().lower()
    try:
        binv = int(r["binary"])
    except:
        binv = -1

    # 1) superclase válida
    if sc and sc not in allowed_super:
        issues.append(("superclass_not_allowed", i, lab, sc, binv))

    # 2) incoherencias clínico-binarias
    if any_match(lab, MALIG) and binv != 1:
        issues.append(("should_be_malignant", i, lab, sc, binv))
    if any_match(lab, PRECANCER_POS) and binv != 1:
        issues.append(("akiec_should_be_1", i, lab, sc, binv))
    if any_match(lab, BENIGN) and binv != 0:
        issues.append(("should_be_benign", i, lab, sc, binv))
    if any_match(lab, AMBIG):
        issues.append(("ambiguous_needs_manual_qc", i, lab, sc, binv))

# Resumen
print("Total rows:", len(df))
print("Issues found:", len(issues))
for tag, idx, lab, sc, binv in issues[:100]:
    print(f"[{tag}] row={idx} label='{lab}' superclass='{sc}' binary={binv}")

