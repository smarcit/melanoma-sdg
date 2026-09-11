#!/usr/bin/env python3
"""Reproducible FID/KID pipeline: real ISIC2019 melanoma test set vs. synthetic stages.

Sampling, preprocessing and metric computation are deterministic given (sample_size, seed).
CPU only. See --help for CLI options.

--sample-size and the e4_synth stage: the e4_synth candidate pool is always restricted to
images with used_in_E4_SYNTH == 1 (currently 1293). --sample-size only chooses how many of
those images are drawn (deterministic subsample via random.Random(seed), no replacement) for
smaller/smoke runs; it never admits images outside that pool. Requesting more than the pool
size is a hard error. This does not redefine the final E4_SYNTH set, only how much of it a
given run materializes.
"""
import argparse
import csv
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, UnidentifiedImageError

RESIZE_POLICY = "short_side_256_center_crop_256_rgb"
TARGET_SIZE = 256
FEATURE_EXTRACTOR = "inception-v3-compat"

STAGE_ORDER = ["raw", "filtered", "accepted", "e4_synth"]
STAGE_CHOICES = STAGE_ORDER + ["all"]

STAGE_TO_GROUP = {
    "raw": "synth_raw",
    "filtered": "synth_filtered",
    "accepted": "synth_accepted",
    "e4_synth": "synth_e4_synth",
}

OUTPUT_COLUMNS = [
    "stage", "n_real", "n_synth", "fid", "kid_mean", "kid_std",
    "sample_size", "seed", "resize_policy", "device", "status",
    "n_real_requested", "n_synth_requested", "n_real_failed", "n_synth_failed",
    "feature_extractor", "batch_size", "num_workers", "error", "timestamp",
]

MANIFEST_COLUMNS = ["group", "filename", "source_path", "seed", "sample_size"]


@dataclass
class Candidate:
    key: str
    path: Optional[Path]


def find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "data").is_dir() and (candidate / "reports").is_dir():
            return candidate
    raise RuntimeError(f"Could not auto-locate repo root from {start}; pass --repo-root explicitly.")


def build_index_by_stem(directory: Path) -> dict:
    return {p.stem: p for p in directory.iterdir() if p.is_file()}


def build_index_by_name(directory: Path) -> dict:
    return {p.name: p for p in directory.iterdir() if p.is_file()}


def resolve_real_reference(repo_root: Path):
    csv_path = repo_root / "data/integrated/master_metadata_split_corrected.csv"
    image_dir = repo_root / "data/imagefolder/test/melanoma"
    df = pd.read_csv(csv_path, low_memory=False)
    sub = df[(df["dataset"] == "ISIC2019") & (df["split"] == "test") & (df["superclass"] == "melanoma")]
    index = build_index_by_stem(image_dir)
    return [Candidate(name, index.get(name)) for name in sub["image_name"].tolist()]


def resolve_synth_raw(repo_root: Path):
    csv_path = repo_root / "data/synth_metadata.csv"
    gen_dir = repo_root / "data/gen/SD_E2_HOPE/v1_8000"
    df = pd.read_csv(csv_path, low_memory=False)
    index = build_index_by_name(gen_dir)
    return [Candidate(name, index.get(name)) for name in df["image_name"].tolist()]


def resolve_synth_filtered(repo_root: Path):
    csv_path = repo_root / "data/synth_metadata_filtered2.csv"
    gen_dir = repo_root / "data/gen/SD_E2_HOPE/v1_8000"
    df = pd.read_csv(csv_path, low_memory=False)
    index = build_index_by_name(gen_dir)
    return [Candidate(name, index.get(name)) for name in df["image_name"].tolist()]


def resolve_synth_accepted(repo_root: Path):
    # NOTE: the 'path' column in this CSV is stale (points to a different machine).
    # Only its basename is trustworthy; resolve that basename against the real generated dir.
    csv_path = repo_root / "data/synthetic_melanoma_accepted.csv"
    gen_dir = repo_root / "data/gen/SD_E2_HOPE/v1_8000"
    df = pd.read_csv(csv_path, low_memory=False)
    index = build_index_by_name(gen_dir)
    basenames = [Path(p).name for p in df["path"].tolist()]
    return [Candidate(name, index.get(name)) for name in basenames]


def resolve_synth_e4(repo_root: Path):
    csv_path = repo_root / "reports/results/synthetic_final_traceability.csv"
    gen_dir = repo_root / "data/gen/SD_E2_HOPE/v1_8000"
    df = pd.read_csv(csv_path, low_memory=False)
    sub = df[df["used_in_E4_SYNTH"] == 1]
    index = build_index_by_name(gen_dir)
    return [Candidate(name, index.get(name)) for name in sub["image_name"].tolist()]


GROUP_RESOLVERS = {
    "real_isic_melanoma": resolve_real_reference,
    "synth_raw": resolve_synth_raw,
    "synth_filtered": resolve_synth_filtered,
    "synth_accepted": resolve_synth_accepted,
    "synth_e4_synth": resolve_synth_e4,
}


def deterministic_sample(candidates, sample_size: int, seed: int, exact: bool = False):
    """Sample without replacement from a lexicographically sorted key list (order-independent, reproducible)."""
    by_key = {c.key: c for c in candidates}
    keys = sorted(by_key.keys())
    if exact:
        chosen = keys
    else:
        if sample_size > len(keys):
            raise RuntimeError(f"requested sample_size={sample_size} exceeds available candidates={len(keys)}")
        chosen = sorted(random.Random(seed).sample(keys, sample_size))
    return [by_key[k] for k in chosen]


def preprocess_image(path: Path, target_size: int = TARGET_SIZE) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w <= h:
            new_w = target_size
            new_h = max(target_size, round(h * target_size / w))
        else:
            new_h = target_size
            new_w = max(target_size, round(w * target_size / h))
        im = im.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        im = im.crop((left, top, left + target_size, top + target_size))
        return im.copy()


def materialize_group(candidates, out_dir: Path, group_label: str, error_log: list) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for c in candidates:
        if c.path is None or not c.path.exists():
            error_log.append((group_label, c.key, "unresolved_or_missing_file"))
            continue
        try:
            img = preprocess_image(c.path)
        except (UnidentifiedImageError, OSError) as exc:
            error_log.append((group_label, c.key, f"open_or_convert_failed:{exc}"))
            continue
        dest = out_dir / f"{Path(c.key).stem}.png"
        img.save(dest, format="PNG")
        ok += 1
    return ok


def run_stage_metrics(real_dir: Path, synth_dir: Path, seed: int, batch_size: int, num_workers: int) -> dict:
    from torch_fidelity import calculate_metrics

    # torch-fidelity 0.3.0 defaults kid_subset_size to 1000 and errors out if either input
    # has fewer samples than that. Cap it to the smaller of the two group sizes so small
    # (e.g. smoke-test) runs work; for N >= 1000 (e.g. the full 1293-image run) this is a
    # no-op and the default of 1000 is used, so the experimental protocol is unchanged.
    n_real = sum(1 for p in real_dir.iterdir() if p.is_file())
    n_synth = sum(1 for p in synth_dir.iterdir() if p.is_file())
    kid_subset_size = min(1000, n_real, n_synth)

    metrics = calculate_metrics(
        input1=str(real_dir),
        input2=str(synth_dir),
        cuda=False,
        fid=True,
        kid=True,
        isc=False,
        ppl=False,
        verbose=True,
        batch_size=batch_size,
        samples_find_deep=False,
        feature_extractor=FEATURE_EXTRACTOR,
        dataloader_num_workers=num_workers,
        rng_seed=seed,
        kid_subset_size=kid_subset_size,
    )
    return {
        "fid": float(metrics["frechet_inception_distance"]),
        "kid_mean": float(metrics["kernel_inception_distance_mean"]),
        "kid_std": float(metrics["kernel_inception_distance_std"]),
    }


def build_manifest_rows(group: str, candidates, seed: int, sample_size: int) -> list:
    """Record the exact (pre-materialization) source files selected for a group."""
    return [
        {
            "group": group,
            "filename": c.key,
            "source_path": str(c.path) if c.path is not None else "",
            "seed": seed,
            "sample_size": sample_size,
        }
        for c in candidates
    ]


def write_manifest(manifest_path: Path, rows: list) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def validate_selections(selections: dict, expected_count: int, e4_full_keys: Optional[set]) -> list:
    """selections: group_name -> list[Candidate]. Returns a list of human-readable problems (empty if all OK)."""
    problems = []
    for group, candidates in selections.items():
        keys = [c.key for c in candidates]
        if len(keys) != expected_count:
            problems.append(f"{group}: expected {expected_count} images, got {len(keys)}")
        if len(set(keys)) != len(keys):
            problems.append(f"{group}: duplicate filenames detected within group")
        paths = [str(c.path) for c in candidates if c.path is not None]
        if len(set(paths)) != len(paths):
            problems.append(f"{group}: duplicate source paths detected within group")
        missing = [c.key for c in candidates if c.path is None or not c.path.exists()]
        if missing:
            problems.append(f"{group}: {len(missing)} missing/unresolved file(s), e.g. {missing[:3]}")
    if e4_full_keys is not None and "synth_e4_synth" in selections:
        e4_keys = {c.key for c in selections["synth_e4_synth"]}
        if e4_keys != e4_full_keys:
            problems.append(
                "synth_e4_synth: selected set does not exactly match the used_in_E4_SYNTH==1 pool"
            )
    return problems


def append_result_row(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None, help="Repo root (auto-detected if omitted).")
    parser.add_argument("--sample-size", type=int, default=1293)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--stages", nargs="+", choices=STAGE_CHOICES, default=["all"])
    parser.add_argument("--output", type=Path, default=None, help="Default: reports/results/synthetic_fid_kid.csv")
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path(__file__).resolve())

    output_path = args.output if args.output else repo_root / "reports/results/synthetic_fid_kid.csv"
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    owns_temp_dir = args.temp_dir is None
    temp_dir = args.temp_dir if args.temp_dir else Path(tempfile.mkdtemp(prefix="fid_kid_"))
    temp_dir.mkdir(parents=True, exist_ok=True)

    stages = STAGE_ORDER if "all" in args.stages else args.stages
    timestamp = datetime.now(timezone.utc).isoformat()
    errors = []
    exit_code = 0

    try:
        real_candidates = resolve_real_reference(repo_root)
        real_selected = deterministic_sample(real_candidates, args.sample_size, args.seed)

        # Resolve + sample every group up front (before any materialization) so the
        # manifest records exactly what was selected, and so pre-flight validation can
        # run before any metric computation.
        stage_selected = {}
        stage_target = {}
        for stage in stages:
            group_name = STAGE_TO_GROUP[stage]
            candidates = GROUP_RESOLVERS[group_name](repo_root)
            if stage == "e4_synth":
                # candidates here is already restricted to used_in_E4_SYNTH == 1 (see
                # resolve_synth_e4). --sample-size only controls how many of *those*
                # images are drawn for a smoke test / smaller run; it can never pull in
                # images outside the final E4_SYNTH set.
                n_e4 = len(candidates)
                if args.sample_size > n_e4:
                    raise RuntimeError(
                        f"E4_SYNTH only contains {n_e4} images (used_in_E4_SYNTH==1); "
                        f"requested sample_size={args.sample_size} exceeds this pool."
                    )
                exact = args.sample_size == n_e4
                target = n_e4 if exact else args.sample_size
            else:
                exact = False
                target = args.sample_size
            stage_selected[group_name] = deterministic_sample(candidates, target, args.seed, exact=exact)
            stage_target[group_name] = target

        manifest_rows = build_manifest_rows("real_isic_melanoma", real_selected, args.seed, args.sample_size)
        for group_name, selected in stage_selected.items():
            manifest_rows.extend(build_manifest_rows(group_name, selected, args.seed, args.sample_size))
        manifest_path = repo_root / "reports/results/synthetic_fid_kid_manifest.csv"
        write_manifest(manifest_path, manifest_rows)

        e4_full_keys = None
        if "synth_e4_synth" in stage_selected:
            e4_full_candidates = GROUP_RESOLVERS["synth_e4_synth"](repo_root)
            e4_full_keys = {c.key for c in e4_full_candidates}

        selections_for_validation = {"real_isic_melanoma": real_selected, **stage_selected}
        problems = validate_selections(selections_for_validation, args.sample_size, e4_full_keys)
        if problems:
            raise RuntimeError("pre-flight validation failed: " + "; ".join(problems))

        real_dir = temp_dir / "real_isic_melanoma"
        n_real_ok = materialize_group(real_selected, real_dir, "real_isic_melanoma", errors)
        if n_real_ok < args.sample_size:
            raise RuntimeError(f"real reference materialization failed: only {n_real_ok}/{args.sample_size} usable images")

        for stage in stages:
            group_name = STAGE_TO_GROUP[stage]
            target = stage_target[group_name]
            row = {
                "stage": stage,
                "sample_size": args.sample_size,
                "seed": args.seed,
                "resize_policy": RESIZE_POLICY,
                "device": "cpu",
                "feature_extractor": FEATURE_EXTRACTOR,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "timestamp": timestamp,
                "n_real": n_real_ok,
                "n_real_requested": args.sample_size,
                "n_real_failed": 0,
            }
            try:
                selected = stage_selected[group_name]
                synth_dir = temp_dir / group_name
                stage_errors = []
                n_synth_ok = materialize_group(selected, synth_dir, group_name, stage_errors)
                errors.extend(stage_errors)
                if n_synth_ok < target:
                    raise RuntimeError(f"{stage} materialization failed: only {n_synth_ok}/{target} usable images")

                metrics = run_stage_metrics(real_dir, synth_dir, args.seed, args.batch_size, args.num_workers)
                row.update({
                    "n_synth": n_synth_ok,
                    "n_synth_requested": target,
                    "n_synth_failed": len(stage_errors),
                    "fid": metrics["fid"],
                    "kid_mean": metrics["kid_mean"],
                    "kid_std": metrics["kid_std"],
                    "status": "ok",
                    "error": "",
                })
            except Exception as exc:
                exit_code = 1
                row.update({
                    "n_synth": row.get("n_synth", ""),
                    "n_synth_requested": row.get("n_synth_requested", ""),
                    "n_synth_failed": row.get("n_synth_failed", ""),
                    "fid": "", "kid_mean": "", "kid_std": "",
                    "status": "failed", "error": str(exc),
                })
                print(f"[ERROR] stage={stage}: {exc}", file=sys.stderr)
            append_result_row(output_path, row)
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        elif owns_temp_dir:
            print(f"[info] temp dir kept at: {temp_dir}", file=sys.stderr)

    if errors:
        print(f"[warn] {len(errors)} image(s) failed to resolve/process across all stages.", file=sys.stderr)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
