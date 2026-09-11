# ===== paths & flags =====
MASTER_CSV      = "../../data/integrated/master_metadata_split_corrected.csv"     # must contain ['split','superclass']
VAL_PREDS_CSV   = "../eval/E2_cand/preds_val.csv"                 # optional: per-sample predictions (val)
USE_VAL_PREDS   = True                                    # set False if you don't have preds
OUT_DIR         = "./data/reports/balance_graphs"                  # outputs (csv/json/plots)

# ===== imports =====
import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== prepare out dir =====
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ===== Read metadata =====
df = pd.read_csv(MASTER_CSV, low_memory=False)
df['superclass'] = df['superclass'].astype(str).str.strip().fillna("other")
df['split'] = df['split'].astype(str).str.strip()

df_tr = df[df['split'] == 'train'].copy()
df_va = df[df['split'] == 'val'].copy()

# ===== Classes & counts =====
classes = sorted([c for c in df_tr['superclass'].unique() if c.lower() != 'nan'])
cnt_tr = df_tr['superclass'].value_counts().reindex(classes).fillna(0).astype(int)
cnt_va = df_va['superclass'].value_counts().reindex(classes).fillna(0).astype(int)

# ===== Compute imbalance metrics =====
# Max/Min ratio, entropy, Gini, coefficient of variation
N_tr = int(cnt_tr.sum())
max_cnt = int(cnt_tr.max())
min_cnt = int(cnt_tr.min())
imbalance_ratio = (max_cnt / max(1, min_cnt))
p = (cnt_tr / max(1, N_tr)).values.astype(float)
entropy = float(-(p * np.log(p + 1e-12)).sum())
gini = float(1.0 - (p**2).sum())
cv = float(cnt_tr.std(ddof=0) / (cnt_tr.mean() + 1e-9))

imbalance_tbl = pd.DataFrame({
    "class": classes,
    "train_count": cnt_tr.values,
    "val_count": cnt_va.values,
    "freq_train": (cnt_tr / max(1, N_tr)).values
})
imbalance_tbl.to_csv(os.path.join(OUT_DIR, "class_counts.csv"), index=False)

with open(os.path.join(OUT_DIR, "imbalance_summary.json"), "w") as f:
    json.dump({
        "num_classes": int(len(classes)),
        "total_train": N_tr,
        "max_per_class": max_cnt,
        "min_per_class": min_cnt,
        "imbalance_ratio_max_min": float(imbalance_ratio),
        "entropy": entropy,
        "gini": gini,
        "coef_variation": cv
    }, f, indent=2)

# ===== Candidate sampler weights =====
# Inverse-Frequency (normalized ~ C)
w_inv = (1.0 / cnt_tr.replace(0, np.nan)).fillna(0.0)
w_inv = (w_inv / (w_inv.sum() + 1e-12)) * len(classes)
pd.DataFrame({"class": classes, "weight_invfreq": w_inv.values}).to_csv(
    os.path.join(OUT_DIR, "inverse_freq_weights.csv"), index=False
)

# Class-Balanced (Effective Number) sweep over beta
def effective_num(n, beta):
    return (1.0 - (beta ** n)) / (1.0 - beta + 1e-12)

betas = [0.9, 0.95, 0.99, 0.999, 0.9995, 0.9999]
rows = []
for b in betas:
    eff = effective_num(cnt_tr.values.astype(float), b)
    w_cb = 1.0 / eff
    w_cb = w_cb / (w_cb.sum() + 1e-12) * len(classes)
    rows.append(pd.DataFrame({"class": classes, "beta": b, "weight": w_cb}))
cb_df = pd.concat(rows, ignore_index=True)
cb_df.to_csv(os.path.join(OUT_DIR, "class_balanced_weights_beta_sweep.csv"), index=False)

# ===== Plots: counts & weights sweep =====
# Plot class counts (train)
plt.figure(figsize=(9, 4))
plt.bar(range(len(classes)), cnt_tr.values)
plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
plt.ylabel("Train count")
plt.title("Class distribution (train)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_class_counts.png"))
plt.close()

# Plot weight curves across betas + inverse-freq
plt.figure(figsize=(10, 5))
for b in betas:
    sub = cb_df[cb_df["beta"] == b]
    plt.plot(range(len(classes)), sub["weight"].values, marker="o", label=f"beta={b}")
plt.plot(range(len(classes)), w_inv.values, linestyle="--", color="k", label="inverse-freq")
plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
plt.ylabel("Normalized weight")
plt.title("Class weights: Inverse-Freq vs Class-Balanced (beta sweep)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_weights_sweep.png"))
plt.close()

# ===== Optional: per-class performance vs count (from val preds) =====
# Expect columns in preds_val.csv: 'target_idx', 'pred_idx' (and optionally prob_* for AUROC later)
perf_tbl = None
if USE_VAL_PREDS and os.path.exists(VAL_PREDS_CSV):
    pv = pd.read_csv(VAL_PREDS_CSV)
    if {"target_idx", "pred_idx"}.issubset(pv.columns):
        C = len(classes)
        f1_per, rec_per = [], []
        for c in range(C):
            y_true = (pv["target_idx"].values == c).astype(int)
            y_pred = (pv["pred_idx"].values == c).astype(int)
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            f1_per.append(float(f1))
            rec_per.append(float(rec))
        perf_tbl = pd.DataFrame({
            "class": classes,
            "train_count": cnt_tr.values,
            "val_count": cnt_va.values,
            "recall_val": rec_per,
            "f1_val": f1_per
        })
        perf_tbl.to_csv(os.path.join(OUT_DIR, "per_class_perf_vs_counts.csv"), index=False)

        # Plot recall vs count (log scale)
        plt.figure(figsize=(6, 4))
        plt.scatter(cnt_tr.values, rec_per)
        for i, cname in enumerate(classes):
            plt.annotate(cname, (cnt_tr.values[i], rec_per[i]), fontsize=8, xytext=(2, 2), textcoords="offset points")
        plt.xscale("log")
        plt.xlabel("Train count (log)")
        plt.ylabel("Recall on val")
        plt.title("Per-class recall vs train count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "plot_recall_vs_count.png"))
        plt.close()

# ===== Focal gamma guideline (heuristic) =====
# If we have per-class recall, use difficulty = 1 - recall; else proxy by 1/sqrt(count)
gammas = [1.0, 1.5, 2.0, 2.5]
if perf_tbl is not None and "recall_val" in perf_tbl.columns:
    diff = 1.0 - np.clip(perf_tbl["recall_val"].values.astype(float), 0.0, 1.0)
else:
    diff = 1.0 / np.sqrt(cnt_tr.values.astype(float) + 1e-9)
diff = diff / (diff.max() + 1e-9)

rows = []
for g in gammas:
    score = diff ** g
    score = score / (score.sum() + 1e-12) * len(classes)
    rows.append(pd.DataFrame({"class": classes, "gamma": g, "importance_score": score}))
focal_df = pd.concat(rows, ignore_index=True)
focal_df.to_csv(os.path.join(OUT_DIR, "focal_gamma_guideline.csv"), index=False)

# ===== README helper =====
with open(os.path.join(OUT_DIR, "README_balance_decision.txt"), "w") as f:
    f.write(
        "Diagnostics for class balancing decisions:\n"
        "- class_counts.csv: train/val counts per class\n"
        "- imbalance_summary.json: entropy, gini, CV, max/min ratio\n"
        "- inverse_freq_weights.csv: weights for WeightedRandomSampler\n"
        "- class_balanced_weights_beta_sweep.csv: CB weights across betas; pick beta so minority weights are ~2-5x major-class\n"
        "- focal_gamma_guideline.csv: heuristic importance vs gamma in [1.0, 2.5]\n"
        "- plot_class_counts.png: long-tail visualization\n"
        "- plot_weights_sweep.png: inverse-freq vs CB across betas\n"
        "- plot_recall_vs_count.png: (if preds provided) recall vs frequency\n"
    )
