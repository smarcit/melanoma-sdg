# ====== paths & config ======
CHECKPOINT_PATH = "../train/runs/E4_SYNTH/best.pt"   # <- ajusta si usas otro run
DATA_ROOT       = "../../data/imagefolder"                # <- tu raíz ImageFolder
SPLITS          = ["val", "test", "test_fairness"]  # eval order
MODEL_NAME      = "tf_efficientnetv2_s.in1k"
NUM_CLASSES     = 8
IMG_SIZE        = 384
BATCH_SIZE      = 16
NUM_WORKERS     = 8
USE_AMP         = True
OUT_DIR         = "../eval/E4_synth"                      # outputs (CSVs/metrics)

# Optional: master metadata for group-wise analysis (fairness, etc.)
MASTER_CSV      = "../../data/integrated/master_metadata_split_corrected.csv"  # set to "" to disable
GROUP_COL       = ""  # e.g., "fitz_group" or "source"; keep "" to skip group metrics

# ====== imports ======
import os, json, math, torch, timm, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ====== reproducibility ======
torch.manual_seed(27); torch.cuda.manual_seed_all(27); np.random.seed(27)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
device = "cuda" if torch.cuda.is_available() else "cpu"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ====== build model (same head as training) ======
# Create model with correct num_classes to match checkpoint head
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
state = ckpt.get("state_dict", ckpt)
# Strip "module." if present (DDP checkpoints)
state = {k.replace("module.", ""): v for k,v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)  # allow ema/bn buffers diffs
model.to(device)
model.eval()

# ====== transforms (match validation preprocessing) ======
# Use center crop at target size and ImageNet normalization
# Note: timm/validate shows (mean,std) = (0.485,0.456,0.406)/(0.229,0.224,0.225) for ImageNet
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

# ====== helper arrays (class names from folder order) ======
# Read class names per split dynamically (ImageFolder ensures consistent idx->class)
split_to_classes = {}

# ====== evaluation loop over splits ======
for split in SPLITS:
    split_dir = os.path.join(DATA_ROOT, split)
    if not os.path.isdir(split_dir):
        continue  # silent skip if a split doesn't exist

    # Build dataset & loader
    ds = datasets.ImageFolder(root=split_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    split_to_classes[split] = ds.classes

    # Storage
    all_probs = []
    all_targets = []
    all_paths = [p for p,_ in ds.samples]

    # Inference
    # Note: use AMP for speed (consistent with training flags)
    if USE_AMP:
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device=="cuda" else torch.autocast("cpu")
    else:
        class _NoOp: 
            def __enter__(self): pass
            def __exit__(self, *args): pass
        amp_ctx = _NoOp()

    with torch.no_grad(), amp_ctx:
        for x,y in tqdm(dl, desc=f"Eval {split}", leave=False):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())

    # Concatenate results
    P = np.concatenate(all_probs, axis=0)  # (N,C)
    Y = np.concatenate(all_targets, axis=0)  # (N,)

    # ====== metrics (AUROC, F1, ECE) ======
    # AUROC macro/micro and per-class (One-vs-Rest). Handle classes absent in Y gracefully.
    try:
        Y_oh = np.eye(NUM_CLASSES)[Y]
        auroc_macro = roc_auc_score(Y_oh, P, average="macro", multi_class="ovr")
        auroc_micro = roc_auc_score(Y_oh, P, average="micro", multi_class="ovr")
    except Exception:
        auroc_macro, auroc_micro = float("nan"), float("nan")

    auroc_per_class = []
    for c in range(NUM_CLASSES):
        try:
            y_true_c = (Y == c).astype(int)
            y_score_c = P[:, c]
            au_c = roc_auc_score(y_true_c, y_score_c)
        except Exception:
            au_c = float("nan")
        auroc_per_class.append(au_c)

    preds = P.argmax(axis=1)
    f1_macro = f1_score(Y, preds, average="macro")
    f1_per_class = f1_score(Y, preds, average=None, labels=list(range(NUM_CLASSES)))

    # Compute ECE (Expected Calibration Error) with 15 bins
    confidences = P.max(axis=1)
    correctness = (preds == Y).astype(float)
    bins = np.linspace(0,1,16)
    ece = 0.0
    for i in range(len(bins)-1):
        m = (confidences > bins[i]) & (confidences <= bins[i+1])
        if m.any():
            acc_b = correctness[m].mean()
            conf_b = confidences[m].mean()
            ece += m.mean() * abs(acc_b - conf_b)

    # ====== save per-sample predictions ======
    # # Save predictions CSV (one row per image)
    out_csv = os.path.join(OUT_DIR, f"preds_{split}.csv")
    df = pd.DataFrame({
        "path": all_paths,
        "target_idx": Y,
        "pred_idx": preds,
        "confidence": confidences
    })
    # Append per-class probabilities as columns
    for c in range(NUM_CLASSES):
        df[f"prob_{c}"] = P[:, c]
    df.to_csv(out_csv, index=False)

    # ====== save metrics JSON ======
    out_json = os.path.join(OUT_DIR, f"metrics_{split}.json")
    metrics = {
        "split": split,
        "classes": ds.classes,
        "auroc_macro": float(auroc_macro),
        "auroc_micro": float(auroc_micro),
        "auroc_per_class": [float(x) if not (isinstance(x,float) and math.isnan(x)) else None for x in auroc_per_class],
        "f1_macro": float(f1_macro),
        "f1_per_class": [float(x) for x in f1_per_class],
        "ece": float(ece),
        "n_samples": int(P.shape[0]),
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "model": MODEL_NAME,
        "checkpoint": CHECKPOINT_PATH
    }
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

# ====== optional: group-wise metrics via master metadata (fairness, domain, etc.) ======
# Read master CSV and compute the same metrics per group if GROUP_COL is provided.
if MASTER_CSV and GROUP_COL:
    meta = pd.read_csv(MASTER_CSV)
    all_preds = []
    for split in SPLITS:
        preds_csv = os.path.join(OUT_DIR, f"preds_{split}.csv")
        if not os.path.isfile(preds_csv):
            continue
        dfp = pd.read_csv(preds_csv)
        dfp["split"] = split
        all_preds.append(dfp)
    if len(all_preds):
        pred_all = pd.concat(all_preds, ignore_index=True)
        meta_sub = meta[[ "path", GROUP_COL ]].copy()
        j = pred_all.merge(meta_sub, on="path", how="left")
        for g, gdf in j.groupby(GROUP_COL):
            Yg = gdf["target_idx"].to_numpy()
            Pg = gdf[[c for c in gdf.columns if c.startswith("prob_")]].to_numpy()
            try:
                Yg_oh = np.eye(NUM_CLASSES)[Yg]
                au_macro_g = roc_auc_score(Yg_oh, Pg, average="macro", multi_class="ovr")
            except Exception:
                au_macro_g = float("nan")
            preds_g = Pg.argmax(axis=1)
            f1_macro_g = f1_score(Yg, preds_g, average="macro")
            conf_g = Pg.max(axis=1); corr_g = (preds_g==Yg).astype(float)
            bins = np.linspace(0,1,16); ece_g=0.0
            for i in range(len(bins)-1):
                m=(conf_g>bins[i])&(conf_g<=bins[i+1])
                if m.any(): ece_g += m.mean()*abs(corr_g[m].mean()-conf_g[m].mean())
            out_json = os.path.join(OUT_DIR, f"metrics_group_{g}.json")
            with open(out_json,"w") as f:
                json.dump({"group": str(g), "auroc_macro": float(au_macro_g), "f1_macro": float(f1_macro_g), "ece": float(ece_g), "n": int(len(Yg))}, f, indent=2)

