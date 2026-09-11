#!/usr/bin/env python3
# Rename Fitzpatrick17k images with query-style URLs (img?imageId=XXXX) to <imageId>.jpg
# and emit a master-like CSV homogeneous to the one already generated

from pathlib import Path
import pandas as pd
from urllib.parse import urlparse, parse_qs
from PIL import Image
import shutil
import string

# Absolute roots and paths
linux_root = Path("/home/marsi/TFM")
images_dir = linux_root / "data/ogs/Fitzpatrick17k/images"
meta_csv = Path("../../data/ogs/Fitzpatrick17k/fitzpatrick17k_metadata.csv")
tax_csv  = Path("../../data/taxonomy_map_fitzpatrick17k.csv")
out_csv = Path("../../data/Fitzpatrick17k_master_like2.csv")

# Read inputs
df  = pd.read_csv(meta_csv, low_memory=False)
tax = pd.read_csv(tax_csv)
tax["k"] = tax["original_label"].astype(str).str.strip().str.lower()
tax_map = {k: (row["superclass"], int(row["binary"])) for k, row in tax.set_index("k").iterrows()}

# Build md5 -> file index (current files still named as md5)
hexdigits = set(string.hexdigits)
by_md5 = {}
for p in images_dir.rglob("*"):
    if not p.is_file():
        continue
    stem = p.stem.lower()
    if len(stem) == 32 and all(ch in hexdigits for ch in stem):
        by_md5[stem] = p

# Helpers
def choose_fitz(row):
    a = str(row.get("fitzpatrick_scale", "")).strip()
    b = str(row.get("fitzpatrick_centaur", "")).strip()
    return a if a not in ("", "nan", "None") else b

def image_info(p: Path):
    try:
        with Image.open(p) as im:
            w, h = im.size
            mode = im.mode
        if mode in ("1", "L"):
            ch = 1
        elif "A" in mode:
            ch = 4
        else:
            ch = 3
        isrgb = mode in ("RGB", "RGBA")
        return w, h, ch, isrgb, False
    except Exception:
        return None, None, None, False, True

# Filter rows that have query-style URL with imageId (and are likely still md5-named)
def get_image_id_from_url(u: str):
    pr = urlparse(str(u))
    q = parse_qs(pr.query)
    return q["imageId"][0] if "imageId" in q and len(q["imageId"]) > 0 else ""

rows = []
for _, r in df.iterrows():
    url = str(r.get("url", "")).strip()
    image_id = get_image_id_from_url(url)
    if image_id == "":
        continue  # skip non-query URLs (already handled previously)

    md5name = str(r.get("md5hash", "")).strip().lower()
    src = by_md5.get(md5name, None)
    if src is None:
        # if we cannot find the md5 file, skip (it may have been processed already)
        continue

    # destination name <imageId>.jpg (keep original extension if available)
    ext = src.suffix.lower() if src.suffix else ".jpg"
    dst = images_dir / f"{image_id}{ext}"
    if src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                shutil.move(str(src), str(dst))
            except Exception:
                dst = src

    final_path = dst.resolve()

    # taxonomy mapping
    lab = str(r.get("label", "")).strip()
    sc, binv = tax_map.get(lab.lower(), ("other", 0))

    # fitzpatrick
    fitz = choose_fitz(r)

    # image info
    w, h, ch, isrgb, corrupt = image_info(final_path)

    # append master-like row (same columns/order you use)
    rows.append({
        "dataset": "Fitzpatrick17k",
        "image_name": final_path.name,
        "path": str(final_path),
        "width": w,
        "height": h,
        "original_label": lab,
        "binary": binv,
        "superclass": sc,
        "subclass": "",
        "acquisition_type": "clinical",
        "anatom_site": "",
        "split": "test_fairness",
        "channels": ch,
        "is_rgb": isrgb,
        "is_corrupt": corrupt,
        "age": "",
        "sex": "",
        "fitzpatrick": fitz,
        "diagnosis_method": "",
        "clinical_diagnosis": "",
        "dermoscopic_diagnosis": "",
        "histological_diagnosis": "",
        "asymmetry": "",
        "color_white": "",
        "color_red": "",
        "color_light_brown": "",
        "color_dark_brown": "",
        "color_blue_gray": "",
        "color_black": "",
        "phash": "",
        "license": "CC-BY 4.0"
    })

# Write output CSV for the processed query-style images
out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=[
    "dataset","image_name","path","width","height","original_label","binary","superclass","subclass",
    "acquisition_type","anatom_site","split","channels","is_rgb","is_corrupt","age","sex","fitzpatrick",
    "diagnosis_method","clinical_diagnosis","dermoscopic_diagnosis","histological_diagnosis","asymmetry",
    "color_white","color_red","color_light_brown","color_dark_brown","color_blue_gray","color_black",
    "phash","license"
]).to_csv(out_csv, index=False)