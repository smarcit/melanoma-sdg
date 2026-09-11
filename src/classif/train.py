# ===== imports =====
import os, json, math, random, numpy as np, pandas as pd
from pathlib import Path
import torch, timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.metrics import accuracy_score
import time, sys
import argparse

# ===== paths & config =====
DATA_ROOT       = "../../data/imagefolder"      # ImageFolder with train/val/test...
OUTPUT_DIR      = "../train/runs/E3_real"        # where to save checkpoints/metrics
MODEL_NAME      = "tf_efficientnetv2_s.in1k"
NUM_CLASSES     = 8
IMG_SIZE        = 384                               # 224/320/384
BATCH_SIZE      = 16
EPOCHS          = 50
LR              = 4e-4
WEIGHT_DECAY    = 3e-4
DROP            = 0.3
DROP_PATH       = 0.2
SMOOTHING       = 0.1
USE_AMP         = True
USE_SAMPLER     = False                            # True -> WeightedRandomSampler (CB weights)
LOSS_KIND       = "cb_focal"                        # "cb_ce" or "cb_focal"
BETA_CB         = 0.9997                            # class-balanced beta
GAMMA_FOCAL     = 2                               # focal gamma
SEED            = 27
MINORITY_NAMES  = ["melanoma","dermatofibroma","vascular_lesion","actinic_keratosis_bowen"]
MEAN_STD        = ((0.485,0.456,0.406),(0.229,0.224,0.225))  # ImageNet

_cli = argparse.ArgumentParser()
_cli.add_argument("--data-root", type=str, default=DATA_ROOT)
_cli.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
_cli.add_argument("--use-sampler", type=int, choices=[0,1], default=int(USE_SAMPLER))
args_cli, _ = _cli.parse_known_args()

DATA_ROOT = args_cli.data_root
OUTPUT_DIR = args_cli.output_dir
USE_SAMPLER = bool(args_cli.use_sampler)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# --- early stopping & extra checkpoints ---
EARLY_STOP_PATIENCE = 10   # 0 = disabled; e.g., 7 to enable patience=7
SAVE_EVERY_EPOCHS = 5     # 0 = disabled; else save epoch_k.pt every N epochs

# ===== reproducibility =====
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
device = "cuda" if torch.cuda.is_available() else "cpu"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ===== albumentations pipelines =====
# # CLAHE + geometric/photometric augment (minor stronger than major)
# # Note: CLAHE here works on RGB; it's a practical approximation.
clahe = A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), always_apply=True)

minor_aug = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.5),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
    A.Rotate(limit=40, p=0.5),
    clahe,
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
    A.Normalize(mean=MEAN_STD[0], std=MEAN_STD[1]),
    ToTensorV2()
])

major_aug = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.85, 1.0), ratio=(0.95, 1.05), p=0.3),
    A.HorizontalFlip(p=0.5),
    clahe,
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01, p=0.3),
    A.Normalize(mean=MEAN_STD[0], std=MEAN_STD[1]),
    ToTensorV2()
])

val_tf = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
    clahe,
    A.Normalize(mean=MEAN_STD[0], std=MEAN_STD[1]),
    ToTensorV2()
])

# ===== helper: Albumentations wrapper for ImageFolder =====
class AlbumentationsImageFolder(datasets.ImageFolder):
    # Apply per-class transform: minor vs major
    def __init__(self, root, transform_minor, transform_major, minority_names, eval_mode=False):
        super().__init__(root=root)
        self.eval_mode = eval_mode
        self.t_minor = transform_minor
        self.t_major = transform_major
        self.minority_ids = set([self.class_to_idx[n] for n in minority_names if n in self.class_to_idx])

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = cv2.imread(path)[:, :, ::-1]  # BGR->RGB
        if self.eval_mode:
            aug = val_tf(image=img); x = aug["image"]
        else:
            if target in self.minority_ids:
                aug = self.t_minor(image=img); x = aug["image"]
            else:
                aug = self.t_major(image=img); x = aug["image"]
        return x, target

# ===== datasets =====
train_dir = os.path.join(DATA_ROOT, "train")
val_dir   = os.path.join(DATA_ROOT, "val")

# temp dataset to get class_to_idx and counts
_tmp = datasets.ImageFolder(root=train_dir, transform=transforms.ToTensor())
CLASSES = _tmp.classes
minority_ids = [ _tmp.class_to_idx[n] for n in MINORITY_NAMES if n in _tmp.class_to_idx ]

train_ds = AlbumentationsImageFolder(train_dir, minor_aug, major_aug, MINORITY_NAMES, eval_mode=False)
val_ds   = AlbumentationsImageFolder(val_dir,   minor_aug, major_aug, MINORITY_NAMES, eval_mode=True)

# ===== sampler (optional, Class-Balanced weights) =====
def effective_num(n, beta):
    return (1 - beta**n) / (1 - beta + 1e-12)

if USE_SAMPLER:
    # per-class counts
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    for _, y in _tmp.samples:
        counts[y] += 1
    eff = effective_num(counts.astype(float), BETA_CB)
    class_weights = 1.0 / eff
    class_weights = class_weights / class_weights.sum() * len(CLASSES)
    # per-sample weight
    sample_weights = [class_weights[y] for _, y in _tmp.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
else:
    sampler = None

# ===== loaders =====
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=(sampler is None),
                          sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=max(8, BATCH_SIZE//2), shuffle=False,
                          num_workers=4, pin_memory=True)

# ===== model =====
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES, drop_rate=DROP, drop_path_rate=DROP_PATH)
model = model.to(device)

# ===== loss: CB-CE / CB-Focal =====
# compute class-balanced weights (for the loss), independent of sampler
counts = np.zeros(len(CLASSES), dtype=np.int64)
for _, y in _tmp.samples:
    counts[y] += 1
eff = effective_num(counts.astype(float), BETA_CB)
cb_weights = torch.tensor((1.0/eff) / (1.0/eff).sum() * len(CLASSES), dtype=torch.float32, device=device)

class CB_CE(nn.Module):
    # CrossEntropy with class-balanced weights
    def __init__(self, weights, smoothing=0.0):
        super().__init__()
        self.w = weights
        self.smoothing = smoothing
    def forward(self, logits, target):
        if self.smoothing > 0:
            n = logits.size(1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.smoothing / (n - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
            logp = F.log_softmax(logits, dim=1)
            loss = -(true_dist * logp).sum(dim=1)
        else:
            loss = F.cross_entropy(logits, target, reduction='none')
        w = self.w[target]
        return (loss * w).mean()

class CB_Focal(nn.Module):
    # Focal(gamma) with class-balanced weights
    def __init__(self, weights, gamma=2.0, smoothing=0.0):
        super().__init__()
        self.w = weights
        self.g = gamma
        self.smoothing = smoothing
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        if self.smoothing > 0:
            n = logits.size(1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.smoothing / (n - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
            pt = (p * true_dist).sum(dim=1)
            ce = -(true_dist * logp).sum(dim=1)
        else:
            pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
            ce = F.nll_loss(logp, target, reduction='none')
        focal = ((1 - pt) ** self.g) * ce
        w = self.w[target]
        return (focal * w).mean()

if LOSS_KIND == "cb_ce":
    criterion = CB_CE(cb_weights, smoothing=SMOOTHING)
else:
    criterion = CB_Focal(cb_weights, gamma=GAMMA_FOCAL, smoothing=SMOOTHING)

# ===== optimizer & scaler =====
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
with open(metrics_path, "w") as f:
    f.write("epoch,lr,train_loss,val_top1,best_top1\n")

# ===== train/val loops (minimal) =====
best_acc = 0.0
no_improve = 0
amp_ok = USE_AMP and (device == "cuda")

for epoch in range(EPOCHS):
    t0 = time.time()
    model.train()
    train_loss_sum = 0.0
    n_batches = 0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if amp_ok:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x); loss = criterion(logits, y)
            loss.backward(); optimizer.step()

        train_loss_sum += float(loss.detach().cpu().item())
        n_batches += 1

    avg_train_loss = train_loss_sum / max(1, n_batches)

    # --- validation (top-1) ---
    model.eval()
    preds = []; targets = []

    if amp_ok:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                z = model(x)
                p = z.argmax(1).cpu().numpy().tolist()
                preds += p; targets += y.numpy().tolist()
    else:
        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                z = model(x)
                p = z.argmax(1).cpu().numpy().tolist()
                preds += p; targets += y.numpy().tolist()

    acc = accuracy_score(targets, preds) if len(preds) else 0.0

    improved = acc > best_acc
    if improved:
        best_acc = acc
        no_improve = 0
        torch.save({
            "model": MODEL_NAME,
            "classes": CLASSES,
            "state_dict": model.state_dict(),
            "img_size": IMG_SIZE
        }, os.path.join(OUTPUT_DIR, "best.pt"))
    else:
        no_improve += 1

    # current LR BEFORE stepping (reflects LR used this epoch)
    cur_lr = optimizer.param_groups[0]["lr"]

    # step scheduler for next epoch
    scheduler.step()

    # extra checkpoints
    if SAVE_EVERY_EPOCHS and ((epoch + 1) % SAVE_EVERY_EPOCHS == 0):
        torch.save({
            "model": MODEL_NAME,
            "classes": CLASSES,
            "state_dict": model.state_dict(),
            "img_size": IMG_SIZE
        }, os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}.pt"))

    # early stopping
    if EARLY_STOP_PATIENCE > 0 and no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch + 1} (no improvement for {EARLY_STOP_PATIENCE} epochs)", flush=True)
        break

    dt = time.time() - t0
    print(f"Epoch {epoch+1:03d}/{EPOCHS} | lr={cur_lr:.3e} | train_loss={avg_train_loss:.4f} | "
          f"val_top1={acc:.4f} | best={best_acc:.4f} | {dt:.1f}s", flush=True)

    with open(metrics_path, "a") as f:
        f.write(f"{epoch+1},{cur_lr:.6e},{avg_train_loss:.6f},{acc:.6f},{best_acc:.6f}\n")

# ===== save final checkpoint & summary =====
torch.save({"model": MODEL_NAME,
            "classes": CLASSES,
            "state_dict": model.state_dict(),
            "img_size": IMG_SIZE}, os.path.join(OUTPUT_DIR, "last.pt"))
with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump({"best_val_top1": float(best_acc),
               "epochs": EPOCHS, "img_size": IMG_SIZE,
               "loss": LOSS_KIND, "beta_cb": BETA_CB, "gamma_focal": GAMMA_FOCAL,
               "use_sampler": USE_SAMPLER}, f, indent=2)
