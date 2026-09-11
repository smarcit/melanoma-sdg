# Rehacer metadata.jsonl correcto para datasets "imagefolder"
import os, json

TRAIN_IMG_DIR = "../../data/resized_256/melanoma"
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
OUT_JSONL = os.path.join(TRAIN_IMG_DIR, "metadata.jsonl")

with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for fn in os.listdir(TRAIN_IMG_DIR):
        if not fn.lower().endswith(EXTS):
            continue
        stem = os.path.splitext(fn)[0]
        txtp = os.path.join(TRAIN_IMG_DIR, stem + ".txt")
        if not os.path.isfile(txtp):
            continue
        with open(txtp, "r", encoding="utf-8") as tf:
            caption = tf.read().strip()
        f.write(json.dumps({"file_name": fn, "text": caption}, ensure_ascii=False) + "\n")

print("OK ->", OUT_JSONL)

