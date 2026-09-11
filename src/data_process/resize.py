# ======= paths & config =======
source_dir = "../../data/imagefolder/train/melanoma"  # carpeta con todas las imágenes reales de melanoma
out_root   = "../../data/resized_256"                   # se creará out_root/melanoma/*.png
img_size   = 256
seed       = 27

# ======= imports =======
import random
from pathlib import Path
from PIL import Image

# ======= seed =======
random.seed(seed)

# ======= prepare out dir =======
out_dir = Path(out_root) / "melanoma"
out_dir.mkdir(parents=True, exist_ok=True)

# ======= helper =======
# Center-crop square + resize to 256 + save PNG
def process_one(in_path, out_path):
    im = Image.open(in_path).convert("RGB")
    w, h = im.size
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top  = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
    im = im.resize((img_size, img_size), Image.BICUBIC)  # Resize to 256x256
    im.save(out_path)

# ======= iterate and write =======
src = Path(source_dir)
allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
counter = 0
for p in src.rglob("*"):
    if p.is_file() and p.suffix.lower() in allowed:
        try:
            fn = p.stem + ".png"
            dst = out_dir / fn
            # Avoid accidental overwrite if stems collide
            while dst.exists():
                counter += 1
                dst = out_dir / f"{p.stem}_{counter}.png"
            process_one(p, dst)
        except Exception:
            pass  # skip unreadable images silently

