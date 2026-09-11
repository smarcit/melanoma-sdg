# ===== paths & config (edit these) =====
CHECKPOINT_PATH = "../train/runs/E4_SYNTH/best.pt"      # <- ajusta si usas otro run
DATA_ROOT       = "../../data/imagefolder"                   # root with train/val/test_control
SPLIT_NAME      = "test_control"                               # PH2 split folder name
MODEL_NAME      = "tf_efficientnetv2_s.in1k"
IMG_SIZE        = 384                                         # match training size (e.g., 224/320/384)
BATCH_SIZE      = 8                                            # low to avoid OOM
NUM_WORKERS     = 2
USE_AMP         = True
PIN_MEMORY      = False
OUT_DIR         = "../eval/E4_E4_SYNTH_ph2"

# Global training taxonomy (8 superclasses) in the SAME order used at training time
# Ensure this matches the folder order used to train (ImageFolder order). If unsure, set in alphabetical order and retrace.
TRAIN_CLASSES = [
    "actinic_keratosis_bowen",
    "basal_cell_carcinoma",
    "benign_keratosis",
    "dermatofibroma",
    "melanocytic_nevus",
    "melanoma",
    "other",
    "vascular_lesion",
]

# ===== imports =====
import os, json, math, torch, timm, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===== reproducibility & device =====
torch.manual_seed(27); torch.cuda.manual_seed_all(27); np.random.seed(27)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
device = "cuda" if torch.cuda.is_available() else "cpu"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ===== build model (multiclass head as trained) =====
num_classes = len(TRAIN_CLASSES)
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
state = ckpt.get("state_dict", ckpt)
state = {k.replace("module.", ""): v for k,v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
model.to(device)
model.eval()

# ===== transforms (match validation preprocessing) =====
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

# ===== dataset & loader (PH2) =====
split_dir = os.path.join(DATA_ROOT, SPLIT_NAME)
ds = datasets.ImageFolder(root=split_dir, transform=tfm)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# Map PH2 folders to binary targets: melanoma -> 1, melanocytic_nevus (common/atypical nevus) -> 0
# ds.classes for PH2 should be like ['melanocytic_nevus','melanoma'] after your mapping
mel_classname = "melanoma"

# Index of melanoma in the GLOBAL (training) taxonomy
mel_idx_global = TRAIN_CLASSES.index(mel_classname)

# ===== inference =====
all_probs_mel = []
all_targets_bin = []
all_paths = [p for p,_ in ds.samples]

if USE_AMP:
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device=="cuda" else torch.autocast("cpu")
else:
    class _NoOp:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    amp_ctx = _NoOp()

with torch.no_grad(), amp_ctx:
    for x,y_local in dl:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        P = torch.softmax(logits, dim=1).cpu().numpy()   # (N,8)
        p_mel = P[:, mel_idx_global]                     # melanoma probability from GLOBAL head
        all_probs_mel.append(p_mel)

        # local label to binary target: 1 if class name is 'melanoma', else 0
        y_names = [ds.classes[i] for i in y_local.numpy()]
        y_bin = np.array([1 if n==mel_classname else 0 for n in y_names], dtype=np.int64)
        all_targets_bin.append(y_bin)

Pmel = np.concatenate(all_probs_mel, axis=0)            # shape (N,)
Ybin = np.concatenate(all_targets_bin, axis=0)          # shape (N,)

# Non-melanoma probability (sum of other classes)
Pnon = 1.0 - Pmel

# ===== binary metrics =====
# AUROC (binary)
try:
    auroc = roc_auc_score(Ybin, Pmel)
except Exception:
    auroc = float("nan")

# Threshold 0.5 for discrete metrics
pred_bin = (Pmel >= 0.5).astype(np.int64)
f1_bin = f1_score(Ybin, pred_bin, average="binary")

# Sensitivity / Specificity
tn, fp, fn, tp = confusion_matrix(Ybin, pred_bin, labels=[0,1]).ravel()
sensitivity = tp / (tp + fn) if (tp+fn) > 0 else float("nan")   # TPR
specificity = tn / (tn + fp) if (tn+fp) > 0 else float("nan")   # TNR

# Calibration (ECE) with 15 bins
conf = np.maximum(Pmel, Pnon)
corr = (pred_bin == Ybin).astype(float)
bins = np.linspace(0,1,16)
ece = 0.0
for i in range(len(bins)-1):
    m = (conf > bins[i]) & (conf <= bins[i+1])
    if m.any():
        acc_b = corr[m].mean()
        conf_b = conf[m].mean()
        ece += m.mean() * abs(acc_b - conf_b)

# ===== save outputs =====
preds_csv = os.path.join(OUT_DIR, "preds_test_control_binary.csv")
df = pd.DataFrame({
    "path": all_paths,
    "target_bin": Ybin,
    "pred_bin": pred_bin,
    "p_melanoma": Pmel,
    "p_nonmelanoma": Pnon,
})
df.to_csv(preds_csv, index=False)

metrics_json = os.path.join(OUT_DIR, "metrics_test_control_binary.json")
with open(metrics_json, "w") as f:
    json.dump({
        "split": SPLIT_NAME,
        "taxonomy": "binary_melanoma_vs_nonmelanoma",
        "train_classes_global": TRAIN_CLASSES,
        "mel_idx_global": mel_idx_global,
        "auroc": float(auroc),
        "f1_binary": float(f1_bin),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ece": float(ece),
        "n": int(len(Ybin)),
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "checkpoint": CHECKPOINT_PATH
    }, f, indent=2)

