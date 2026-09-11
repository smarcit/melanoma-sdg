# tb_export.py
# Usage: python tb_export.py /path/to/events.out.tfevents.* out.csv
# Reads a TensorBoard event file and exports all scalar tags to CSV.

import sys, os, csv
from tensorboard.backend.event_processing import event_accumulator

# Read args
evt_path = sys.argv[1]
out_csv  = sys.argv[2] if len(sys.argv) > 2 else "metrics.csv"

# Load event file
ea = event_accumulator.EventAccumulator(
    evt_path,
    size_guidance={
        event_accumulator.SCALARS: 200000,
        event_accumulator.TENSORS: 0,
        event_accumulator.HISTOGRAMS: 0,
        event_accumulator.IMAGES: 0,
        event_accumulator.AUDIO: 0,
        event_accumulator.COMPRESSED_HISTOGRAMS: 0,
    },
)
ea.Reload()

# Collect all scalar tags
scalar_tags = ea.Tags().get("scalars", [])
if not scalar_tags:
    print("No scalar tags found.")
    sys.exit(0)

# Build a dict: step -> { tag: value }
steps = {}
for tag in scalar_tags:
    for ev in ea.Scalars(tag):
        step = ev.step
        val = ev.value
        if step not in steps:
            steps[step] = {}
        steps[step][tag] = val

# Determine header
all_tags_sorted = sorted(scalar_tags)
header = ["step"] + all_tags_sorted

# Write CSV
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    for step in sorted(steps.keys()):
        row = [step] + [steps[step].get(tag, "") for tag in all_tags_sorted]
        w.writerow(row)

print(f"Wrote: {out_csv}")
# Quick summary (best loss if exists)
for cand in ["loss", "train/loss", "training/loss", "loss_total"]:
    if cand in all_tags_sorted:
        best = min((steps[s].get(cand) for s in steps if cand in steps[s]), default=None)
        last_step = max(steps.keys())
        last = steps[last_step].get(cand, None)
        print(f"Loss tag: {cand}  |  best: {best:.6f}  |  last@{last_step}: {last:.6f}")
        break

for cand in ["lr", "learning_rate"]:
    if cand in all_tags_sorted:
        last_step = max(steps.keys())
        last = steps[last_step].get(cand, None)
        print(f"LR tag: {cand}  |  last@{last_step}: {last}")
        break

