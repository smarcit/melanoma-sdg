# make_oversampled_imagefolder.py
from pathlib import Path
import random, shutil, os

SRC = Path("../../data/imagefolder")           # tu dataset real (train/val/test)
DST = Path("../../data/imagefolder_OS")        # dataset sombra con oversampling
N_DUP = 1293                                 # nº de duplicados de melanoma
SEED = 27

def symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst)
    except FileExistsError:
        pass

def mirror_split(split: str):
    src_split = SRC / split
    dst_split = DST / split
    for cls_dir in (d for d in src_split.iterdir() if d.is_dir()):
        out_cls = dst_split / cls_dir.name
        out_cls.mkdir(parents=True, exist_ok=True)
        for img in cls_dir.iterdir():
            if img.is_file():
                symlink(img, out_cls / img.name)

def main():
    # 1) replica train/val tal cual por symlink
    for split in ["train","val"]:
        mirror_split(split)

    # 2) añade N_DUP symlinks extra de melanoma (control "más cantidad, misma diversidad")
    mdir = SRC / "train" / "melanoma"
    mimgs = sorted([p for p in mdir.iterdir() if p.is_file()])
    assert len(mimgs) > 0, "No hay imágenes reales de melanoma en train/melanoma"
    random.seed(SEED)
    picks = [random.choice(mimgs) for _ in range(N_DUP)]
    dst_mdir = DST / "train" / "melanoma"
    for i, p in enumerate(picks):
        symlink(p, dst_mdir / f"{p.stem}__dup{i:05d}{p.suffix}")

    print(f"OK: oversampling → {N_DUP} symlinks añadidos en {dst_mdir}")

if __name__ == "__main__":
    main()

