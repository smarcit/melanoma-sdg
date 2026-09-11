# ===== paths & options =====
MASTER_CSV_IN  = "../../data/integrated/master_metadata_split.csv"         # input master
FITZ_TAXO_CSV  = "../../data/taxonomy_map_fitzpatrick17k.csv"   # original-to-superclass mapping
MASTER_CSV_OUT = "../../data/integrated/master_metadata_split_corrected.csv"

# ===== constants =====
ALLOWED_SUPERCLASSES = [
    "melanoma",
    "melanocytic_nevus",
    "basal_cell_carcinoma",
    "actinic_keratosis_bowen",
    "benign_keratosis",
    "dermatofibroma",
    "vascular_lesion",
    "other",
]

# ===== imports =====
import pandas as pd

# ===== Read metadata =====
df = pd.read_csv(MASTER_CSV_IN)

# ===== Read Fitzpatrick17k taxonomy map =====
# expected columns: e.g., ['original_label','superclass'] (adjust if your header differs)
tax = pd.read_csv(FITZ_TAXO_CSV)

# ===== Fix special cases in taxonomy (SCC and other_malignant) =====
# Map to target superclasses BEFORE merging
tax['superclass'] = tax['superclass'].replace({
    'squamous_cell_carcinoma': 'actinic_keratosis_bowen',   # keratinocytic non-melanoma
    'other_malignant': 'other'                               # residual malignant -> other (non-melanoma)
})

# ===== Merge Fitzpatrick mapping for fairness split =====
# Assumptions:
# - df has columns: 'split', 'dataset', 'original_label' (adjust the column name below to your metadata)
# - mapping only needed for fairness (Fitzpatrick17k); for others we preserve their 'superclass'
df_fair = df[df['split'] == 'test_fairness']
df_others = df[df['split'] != 'test_fairness']

# If your master has a different column name for the fine-grained label used in the mapping, change 'original_label' here
df_fair = df_fair.merge(
    tax[['original_label','superclass']],
    on='original_label', how='left', suffixes=('', '_mapped')
)

# ===== Choose final superclass =====
# For fairness rows: prefer mapped taxonomy if available; otherwise keep existing 'superclass'
df_fair['superclass_final'] = df_fair['superclass_mapped'].where(df_fair['superclass_mapped'].notna(), df_fair['superclass'])

# For non-fairness: keep existing superclass
df_others['superclass_final'] = df_others['superclass']

# ===== Concat back =====
df_all = pd.concat([df_others, df_fair], axis=0, ignore_index=True)

# ===== Normalize to 8 superclasses =====
# Remove stray/NaN and collapse to allowed set
df_all['superclass_final'] = df_all['superclass_final'].fillna('other')
df_all.loc[~df_all['superclass_final'].isin(ALLOWED_SUPERCLASSES), 'superclass_final'] = 'other'

# ===== Binary mapping (melanoma vs non-melanoma) =====
df_all['is_melanoma'] = (df_all['superclass_final'] == 'melanoma').astype(int)

# ===== Overwrite main superclass column and keep shape =====
df_all['superclass'] = df_all['superclass_final']
df_all = df_all.drop(columns=['superclass_final'])
if 'superclass_mapped' in df_all.columns:
    df_all = df_all.drop(columns=['superclass_mapped'])

# ===== Write corrected master =====
df_all.to_csv(MASTER_CSV_OUT, index=False)

