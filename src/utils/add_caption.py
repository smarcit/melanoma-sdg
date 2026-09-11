# ============================
# Build clinical captions for SD-LoRA (DERM12345 + HAM10000, melanoma only)
# ============================
# Paths (edit these)
CSV_PATH = "../../data/integrated/master_metadata_split_corrected.csv"
TRAIN_IMG_DIR = "../../data/resized_256/melanoma"         # folder with 256x256 images used to train SD-LoRA
JSONL_OUT = "../../data/resized_256/melanoma/metadata.jsonl"

# ----------------------------
# Imports
# ----------------------------
# Read metadata
import os, json
import pandas as pd

# ----------------------------
# Load metadata
# ----------------------------
df = pd.read_csv(CSV_PATH)

# ----------------------------
# Minimal normalization helpers (kept inline to keep script flat)
# ----------------------------
# Map common acral indications
ACRAL_TOKENS = {"palm", "palms", "sole", "soles", "palms/soles", "acral"}

# Accept these image extensions
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ----------------------------
# Filter to target cohort
# ----------------------------
# Keep only DERM12345 + HAM10000, melanoma class
mask = (
    df["binary"].astype(str).isin(["1", "1.0", "true", "True"]) &
    df["superclass"].str.lower().eq("melanoma") &
    df["dataset"].str.upper().isin(["DERM12345", "HAM10000"])
)
df = df.loc[mask].copy()

# ----------------------------
# Build a lookup keyed by filename stem (no extension)
# Assumes metadata has a column with filename or path. We try common names.
# ----------------------------
# Try to find the most plausible column for filename
candidate_cols = [c for c in df.columns if c.lower() in ["filename","image","image_name","file","filepath","path"]]
if len(candidate_cols) == 0:
    # Fall back: try to infer from an 'id' column by string cast
    if "id" in df.columns:
        df["_fname"] = df["id"].astype(str)
    else:
        # last resort: assume there is a column 'name' with base name
        df["_fname"] = df.iloc[:,0].astype(str)
else:
    col = candidate_cols[0]
    # Extract base name without extension
    df["_fname"] = df[col].astype(str).apply(lambda s: os.path.splitext(os.path.basename(s))[0])

# ----------------------------
# Caption builder
# ----------------------------
# Normalize strings a bit
def norm_txt(x):
    if pd.isna(x):
        return None
    s = str(x).strip().replace("_"," ").lower()
    return s if s else None

def norm_sex(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ["male","m","masculino","hombre"]:
        return "male"
    if s in ["female","f","femenino","mujer"]:
        return "female"
    return None

def age_phrase(x):
    # Build "55-year-old" if age is a valid positive int
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        n = int(float(s))
        if n > 0:
            return f"{n}-year-old"
    except:
        return None
    return None

# Build per-row caption components
captions = {}
for _, r in df.iterrows():
    fname = str(r["_fname"])

    dataset = str(r.get("dataset","")).upper()
    acquisition_type = norm_txt(r.get("acquisition_type",""))
    anatom_site = norm_txt(r.get("anatom_site",""))
    subclass = norm_txt(r.get("subclass",""))

    # NEW: read age/sex if available (HAM10000 has 100% coverage)
    age_str = age_phrase(r.get("age", None))
    sex_str = norm_sex(r.get("sex", None))

    # Base modality (all dermoscopic in your note)
    modality = "dermoscopic image"
    if acquisition_type and "dermo" in acquisition_type:
        modality = "dermoscopic image"

    # Site clause
    site_clause = ""
    if anatom_site:
        # Mark acral explicitly if site suggests palms/soles
        is_acral = any(tok in anatom_site for tok in ACRAL_TOKENS)
        if is_acral:
            site_clause = ", located on an acral site (palms/soles)"
        else:
            site_clause = f", located on the {anatom_site}"

    # Subtype clause (DERM12345 only typically)
    subtype_clause = ""
    if subclass and dataset == "DERM12345":
        # keep raw clinical subtype term
        subtype_clause = f", subtype: {subclass}"

    # NEW: patient clause (age + sex if available)
    patient_clause = ""
    patient_bits = []
    if age_str:
        patient_bits.append(age_str)
    if sex_str:
        patient_bits.append(sex_str)
    if len(patient_bits) > 0:
        patient_clause = ", patient: " + " ".join(patient_bits)

    # Compose caption text
    # Keep it concise, clinical, and consistent
    text = f"Dermoscopic image of a melanoma{site_clause}{subtype_clause}{patient_clause}, clinical dataset style."

    captions[fname] = text

# ----------------------------
# Write .txt sidecar captions for images present in TRAIN_IMG_DIR
# Also build metadata.jsonl entries for those files
# ----------------------------
entries = []
for fn in os.listdir(TRAIN_IMG_DIR):
    if not fn.lower().endswith(EXTS):
        continue
    stem = os.path.splitext(fn)[0]
    if stem in captions:
        # Write sidecar .txt
        txtp = os.path.join(TRAIN_IMG_DIR, stem + ".txt")
        with open(txtp, "w", encoding="utf-8") as f:
            f.write(captions[stem])

        # Add JSONL entry (absolute path recommended for some trainers)
        imgp = os.path.join(TRAIN_IMG_DIR, fn)
        entries.append({"image": imgp, "text": captions[stem]})

# ----------------------------
# Save metadata.jsonl (Diffusers-compatible)
# ----------------------------
with open(JSONL_OUT, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")