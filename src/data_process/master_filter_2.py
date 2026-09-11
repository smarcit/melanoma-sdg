#!/usr/bin/env python3
# Input and output metadata paths
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

in_csv = Path("../../data/synth_metadata_filtered1_enriched.csv")
out_csv = Path("../../data/synth_metadata_filtered2.csv")

# Read enriched metadata
df = pd.read_csv(in_csv, dtype={"phash": "uint64"})

# Blur threshold as P5
blur_thresh = float(np.percentile(df["blur"].values, 5.0))
df = df.loc[df["blur"] >= blur_thresh].copy()

# Helper columns
df["area"] = (df["width"] * df["height"]).astype(float)
df["keep"] = True
df["dup_group_id"] = ""

# Exact duplicates by md5
gid = 0
for md5, g in df.groupby("md5"):
    if len(g) > 1:
        gid += 1
        group_id = f"G{gid:05d}"
        rep = g.sort_values(["area", "blur"], ascending=[False, False]).index[0]
        df.loc[g.index, "dup_group_id"] = group_id
        df.loc[g.index.difference([rep]), "keep"] = False

# Near-duplicates by pHash (phash stored as uint64)
work = df.loc[df["keep"]].copy()
work["bucket"] = work["phash"].apply(lambda x: format(int(x), "016x")[:4])

idxs = work.index.to_numpy()
paths = work["path"].to_numpy()
areas = work["area"].to_numpy()
blurs = work["blur"].to_numpy()
hash_vals = work["phash"].to_numpy(dtype=np.uint64)
buckets = work["bucket"].to_numpy()

idx_to_group = {}
for b in np.unique(buckets):
    m = buckets == b
    b_idx = idxs[m]
    b_hash = hash_vals[m]
    for i in range(len(b_idx)):
        for j in range(i + 1, len(b_idx)):
            hi = int(b_hash[i]); hj = int(b_hash[j])
            hdist = (hi ^ hj).bit_count()
            if hdist <= 10:
                if hdist <= 5:
                    is_dup = True
                else:
                    im_i = np.array(Image.open(paths[np.where(idxs==b_idx[i])[0][0]]).convert("L"))
                    im_j = np.array(Image.open(paths[np.where(idxs==b_idx[j])[0][0]]).convert("L"))
                    H = min(im_i.shape[0], im_j.shape[0])
                    W = min(im_i.shape[1], im_j.shape[1])
                    sim, _ = ssim(im_i[:H, :W], im_j[:H, :W], full=True)
                    is_dup = (sim >= 0.90)
                if is_dup:
                    gi = idx_to_group.get(b_idx[i], None)
                    gj = idx_to_group.get(b_idx[j], None)
                    if gi is None and gj is None:
                        gid += 1
                        gnew = f"G{gid:05d}"
                        idx_to_group[b_idx[i]] = gnew
                        idx_to_group[b_idx[j]] = gnew
                    elif gi is not None and gj is None:
                        idx_to_group[b_idx[j]] = gi
                    elif gi is None and gj is not None:
                        idx_to_group[b_idx[i]] = gj
                    elif gi != gj:
                        g_keep = min(gi, gj)
                        g_merge = max(gi, gj)
                        for k in list(idx_to_group.keys()):
                            if idx_to_group[k] == g_merge:
                                idx_to_group[k] = g_keep

# Assign near-duplicate groups
for k, v in idx_to_group.items():
    df.loc[k, "dup_group_id"] = v

# Keep best representative inside each group
for gname, g in df.loc[(df["dup_group_id"] != "") & (df["keep"])].groupby("dup_group_id"):
    if len(g) > 1:
        rep = g.sort_values(["area", "blur"], ascending=[False, False]).index[0]
        df.loc[g.index.difference([rep]), "keep"] = False

# Write output
df_out = df.loc[df["keep"]].drop(columns=["bucket", "area", "keep"], errors="ignore")
out_csv.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(out_csv, index=False)