#!/usr/bin/env python3
"""LPIPS-based internal perceptual diversity for five melanoma image groups.

For each group (real_isic_melanoma, synth_raw, synth_filtered, synth_accepted,
synth_e4_synth), LPIPS distances are computed between pairs of DIFFERENT images
sampled from within the SAME group. This is an internal perceptual-diversity
indicator, not a real-vs-synthetic similarity/quality metric.

Groups and images are taken exactly from reports/results/synthetic_fid_kid_manifest.csv
(the canonical FID/KID manifest); this script does not resample the source CSVs.
Preprocessing reuses compute_fid_kid.preprocess_image (RGB, short-side-256 resize,
256x256 center crop). CPU only.
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compute_fid_kid import find_repo_root, preprocess_image, RESIZE_POLICY, TARGET_SIZE

NETWORK = "alex"
DEVICE = "cpu"
EXPECTED_COUNT = 1293

GROUP_ORDER = [
    "real_isic_melanoma",
    "synth_raw",
    "synth_filtered",
    "synth_accepted",
    "synth_e4_synth",
]

PAIRS_COLUMNS = ["group", "image_a", "image_b", "lpips_distance", "seed"]
SUMMARY_COLUMNS = [
    "group", "n_images", "n_pairs", "lpips_mean", "lpips_std", "lpips_median",
    "lpips_min", "lpips_max", "seed", "network", "device", "resize_policy", "status",
]


def load_manifest_groups(manifest_path: Path) -> dict:
    df = pd.read_csv(manifest_path, low_memory=False)
    groups = {}
    for group in GROUP_ORDER:
        sub = df[df["group"] == group].sort_values("filename")
        groups[group] = list(zip(sub["filename"].tolist(), sub["source_path"].tolist()))
    return groups


def validate_groups(groups: dict, expected_count: int) -> list:
    problems = []
    for group, items in groups.items():
        filenames = [f for f, _ in items]
        paths = [p for _, p in items]
        if len(items) != expected_count:
            problems.append(f"{group}: expected {expected_count} images, got {len(items)}")
        if len(set(filenames)) != len(filenames):
            problems.append(f"{group}: duplicate filenames within group")
        if len(set(paths)) != len(paths):
            problems.append(f"{group}: duplicate source paths within group")
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            problems.append(f"{group}: {len(missing)} missing file(s), e.g. {missing[:3]}")
    return problems


def validate_e4_synth(groups: dict, repo_root: Path) -> list:
    """Confirm the manifest's synth_e4_synth set matches used_in_E4_SYNTH==1 exactly."""
    problems = []
    traceability_path = repo_root / "reports/results/synthetic_final_traceability.csv"
    df = pd.read_csv(traceability_path, low_memory=False)
    full_e4_keys = set(df[df["used_in_E4_SYNTH"] == 1]["image_name"].tolist())
    manifest_e4_keys = {f for f, _ in groups["synth_e4_synth"]}
    if manifest_e4_keys != full_e4_keys:
        problems.append("synth_e4_synth: manifest set does not exactly match the used_in_E4_SYNTH==1 pool")
    return problems


def generate_pairs(n: int, num_pairs: int, seed: int) -> list:
    """Deterministic sample of num_pairs unique unordered index pairs (i, j), i < j.

    random.Random(seed).sample(range(n), 2) never returns i == j, so self-pairs are
    impossible by construction. Storing pairs as sorted tuples in a set collapses
    (A, B)/(B, A) duplicates.
    """
    max_pairs = n * (n - 1) // 2
    if num_pairs > max_pairs:
        raise RuntimeError(f"requested num_pairs={num_pairs} exceeds available unique pairs ({max_pairs}) for n={n}")
    rng = random.Random(seed)
    seen = set()
    pairs = []
    while len(pairs) < num_pairs:
        i, j = rng.sample(range(n), 2)
        if i > j:
            i, j = j, i
        if (i, j) in seen:
            continue
        seen.add((i, j))
        pairs.append((i, j))
    return pairs


def validate_pairs(pairs: list, num_pairs: int) -> list:
    problems = []
    if len(pairs) != num_pairs:
        problems.append(f"expected {num_pairs} pairs, got {len(pairs)}")
    if len(set(pairs)) != len(pairs):
        problems.append("duplicate pairs detected")
    self_pairs = [p for p in pairs if p[0] == p[1]]
    if self_pairs:
        problems.append(f"{len(self_pairs)} self-pair(s) detected")
    return problems


def image_to_lpips_tensor(path: str) -> torch.Tensor:
    """RGB, short-side-256 resize + 256x256 center crop (identical to FID/KID),
    then scaled to [-1, 1]: lpips 0.1.4's LPIPS.forward(..., normalize=False)
    (the default) expects inputs already in [-1, 1] (see lpips/lpips.py forward()
    and lpips/__init__.py im2tensor(), which maps uint8 [0, 255] -> /127.5 - 1)."""
    img = preprocess_image(Path(path), TARGET_SIZE)
    arr = np.asarray(img, dtype=np.float32)  # HWC, [0, 255]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
    tensor = tensor / 127.5 - 1.0
    return tensor


def compute_group_distances(model, items: list, pairs: list, batch_size: int) -> list:
    cache = {}

    def get_tensor(idx):
        filename, path = items[idx]
        if filename not in cache:
            cache[filename] = image_to_lpips_tensor(path)
        return cache[filename]

    distances = [None] * len(pairs)
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            batch_a = torch.stack([get_tensor(i) for i, _ in batch_pairs]).to(DEVICE)
            batch_b = torch.stack([get_tensor(j) for _, j in batch_pairs]).to(DEVICE)
            out = model(batch_a, batch_b).view(-1).cpu().numpy()
            for k, d in enumerate(out):
                distances[start + k] = float(d)
    return distances


def verify_pairs_csv_reproducible(pairs_csv_path: Path, groups: dict, seed: int, num_pairs: int) -> bool:
    df = pd.read_csv(pairs_csv_path)
    for group in GROUP_ORDER:
        items = groups[group]
        regen = generate_pairs(len(items), num_pairs, seed)
        expected = [(items[i][0], items[j][0]) for i, j in regen]
        sub = df[df["group"] == group]
        actual = list(zip(sub["image_a"].tolist(), sub["image_b"].tolist()))
        if expected != actual:
            return False
    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None, help="Repo root (auto-detected if omitted).")
    parser.add_argument("--manifest", type=Path, default=None,
                         help="Default: reports/results/synthetic_fid_kid_manifest.csv")
    parser.add_argument("--num-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None, help="Default: reports/results/synthetic_lpips.csv")
    parser.add_argument("--pairs-output", type=Path, default=None,
                         help="Default: reports/results/synthetic_lpips_pairs.csv")
    parser.add_argument("--no-save-pairs", action="store_true", help="Skip writing the per-pair distances CSV.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path(__file__).resolve())

    manifest_path = args.manifest if args.manifest else repo_root / "reports/results/synthetic_fid_kid_manifest.csv"
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    output_path = args.output if args.output else repo_root / "reports/results/synthetic_lpips.csv"
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    pairs_output_path = args.pairs_output if args.pairs_output else repo_root / "reports/results/synthetic_lpips_pairs.csv"
    if not pairs_output_path.is_absolute():
        pairs_output_path = repo_root / pairs_output_path

    print(f"[info] loading manifest: {manifest_path}")
    groups = load_manifest_groups(manifest_path)

    problems = validate_groups(groups, EXPECTED_COUNT)
    problems += validate_e4_synth(groups, repo_root)
    if problems:
        for p in problems:
            print(f"[ERROR] {p}", file=sys.stderr)
        print("[ERROR] dataset validation failed; aborting before LPIPS execution.", file=sys.stderr)
        sys.exit(1)
    print("[info] dataset validation passed for all 5 groups (1293 images, no dup/missing, E4_SYNTH pool intact).")

    group_pairs = {}
    for group, items in groups.items():
        n = len(items)
        pairs_a = generate_pairs(n, args.num_pairs, args.seed)
        pairs_b = generate_pairs(n, args.num_pairs, args.seed)
        if pairs_a != pairs_b:
            print(f"[ERROR] {group}: pair generation is not reproducible for seed={args.seed}", file=sys.stderr)
            sys.exit(1)
        pair_problems = validate_pairs(pairs_a, args.num_pairs)
        if pair_problems:
            for p in pair_problems:
                print(f"[ERROR] {group}: {p}", file=sys.stderr)
            sys.exit(1)
        group_pairs[group] = pairs_a
    print("[info] pair-generation validation passed (uniqueness, no self-pairs, reproducibility) for all 5 groups.")

    import lpips
    print(f"[info] loading LPIPS model (net={NETWORK}, device={DEVICE})...")
    model = lpips.LPIPS(net=NETWORK)
    model.eval()
    model.to(DEVICE)

    pairs_rows = []
    summary_rows = []
    timings = {}
    for group in GROUP_ORDER:
        items = groups[group]
        pairs = group_pairs[group]
        t0 = time.time()
        distances = compute_group_distances(model, items, pairs, args.batch_size)
        timings[group] = time.time() - t0

        finite = [np.isfinite(d) for d in distances]
        status = "ok" if all(finite) else "failed_non_finite"

        for (i, j), d in zip(pairs, distances):
            pairs_rows.append({
                "group": group,
                "image_a": items[i][0],
                "image_b": items[j][0],
                "lpips_distance": d,
                "seed": args.seed,
            })

        arr = np.array(distances, dtype=np.float64)
        summary_rows.append({
            "group": group,
            "n_images": len(items),
            "n_pairs": len(pairs),
            "lpips_mean": float(arr.mean()),
            "lpips_std": float(arr.std()),
            "lpips_median": float(np.median(arr)),
            "lpips_min": float(arr.min()),
            "lpips_max": float(arr.max()),
            "seed": args.seed,
            "network": NETWORK,
            "device": DEVICE,
            "resize_policy": RESIZE_POLICY,
            "status": status,
        })
        print(f"[info] {group}: n_pairs={len(pairs)} mean={arr.mean():.4f} status={status} "
              f"elapsed={timings[group]:.1f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[info] wrote summary: {output_path}")

    reproducible = None
    if not args.no_save_pairs:
        pairs_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pairs_output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PAIRS_COLUMNS)
            writer.writeheader()
            writer.writerows(pairs_rows)
        print(f"[info] wrote pairs: {pairs_output_path}")
        reproducible = verify_pairs_csv_reproducible(pairs_output_path, groups, args.seed, args.num_pairs)
        print(f"[info] post-write reproducibility check (regenerate from seed={args.seed} vs written CSV): "
              f"{'IDENTICAL' if reproducible else 'DIFFERENT'}")

    total_elapsed = sum(timings.values())
    print(f"[info] total LPIPS runtime: {total_elapsed:.1f}s")
    for group, t in timings.items():
        print(f"[info]   {group}: {t:.1f}s")

    any_failed = any(r["status"] != "ok" for r in summary_rows)
    if reproducible is False:
        any_failed = True
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
