# make_synth_master_csv.py
# -*- coding: utf-8 -*-
import csv
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# --------- CONFIG (adapta solo si cambia tu ruta) ----------
INPUT_DIR = Path("/home/marsi/TFM/data/gen/SD_E2_HOPE/v1_8000")
OUTPUT_CSV = Path("/home/marsi/TFM/data/synth_metadata.csv")

# columnas del master_metadata que vamos a rellenar
COLUMNS = [
    "dataset","image_name","path","width","height","original_label","binary","superclass","subclass",
    "acquisition_type","anatom_site","split","channels","is_rgb","is_corrupt","age","sex","fitzpatrick",
    "diagnosis_method","clinical_diagnosis","dermoscopic_diagnosis","histological_diagnosis","asymmetry",
    "color_white","color_red","color_light_brown","color_dark_brown","color_blue_gray","color_black",
    "phash","license","blur","md5"
]

# valores constantes que has pedido
CONST = {
    "dataset": "Synth",
    "width": 284,
    "height": 284,
    "binary": 1,
    "superclass": "melanoma",
    "subclass": "",
    "split": "train",
    "original_label": "",
    "acquisition_type": "",  # razonable por tu pipeline; pon "" si no quieres fijarlo
    "anatom_site": "",
    "age": "",
    "sex": "",
    "fitzpatrick": "",
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
    "license": "",
}

def inspect_image(path: Path):
    """
    Abre con PIL, convierte a RGB para pHash; calcula channels/is_rgb/is_corrupt.
    Para blur, usa BGR (OpenCV).
    """
    try:
        with Image.open(path) as im:
            # canales reales del fichero
            mode = im.mode  # e.g. "RGB", "L", "RGBA"
            # channels: si tiene alfa, lo contamos, pero is_rgb solo si exactamente 3
            channels = {"1":1,"L":1,"P":1,"RGB":3,"RGBA":4,"CMYK":4}.get(mode, len(im.getbands()))
            is_rgb = (mode == "RGB" or channels == 3)
            is_corrupt = False  # si llegamos aquí, abría y convertía bien
            return channels, is_rgb, is_corrupt
    except (UnidentifiedImageError, OSError, ValueError):
        # no se puede abrir o está corrupta
        return 0, False, True

def main():
    rows = []
    img_paths = sorted([p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}])
    if not img_paths:
        raise SystemExit(f"no se encontraron imágenes en {INPUT_DIR}")

    for p in img_paths:
        channels, is_rgb, is_corrupt = inspect_image(p)

        row = {
            "dataset": CONST["dataset"],
            "image_name": p.name,
            "path": str(p),  # p.ej. /home/marsi/TFM/data/gen/SD_E2_HOPE/v1_8000/<imagen>
            "width": CONST["width"],
            "height": CONST["height"],
            "original_label": CONST["original_label"],
            "binary": CONST["binary"],
            "superclass": CONST["superclass"],
            "subclass": CONST["subclass"],
            "acquisition_type": CONST["acquisition_type"],
            "anatom_site": CONST["anatom_site"],
            "split": CONST["split"],
            "channels": channels,
            "is_rgb": int(is_rgb),
            "is_corrupt": int(is_corrupt),
            "age": CONST["age"],
            "sex": CONST["sex"],
            "fitzpatrick": CONST["fitzpatrick"],
            "diagnosis_method": CONST["diagnosis_method"],
            "clinical_diagnosis": CONST["clinical_diagnosis"],
            "dermoscopic_diagnosis": CONST["dermoscopic_diagnosis"],
            "histological_diagnosis": CONST["histological_diagnosis"],
            "asymmetry": CONST["asymmetry"],
            "color_white": CONST["color_white"],
            "color_red": CONST["color_red"],
            "color_light_brown": CONST["color_light_brown"],
            "color_dark_brown": CONST["color_dark_brown"],
            "color_blue_gray": CONST["color_blue_gray"],
            "color_black": CONST["color_black"],
        }
        rows.append(row)

    # escribir CSV
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"ok: {len(rows)} filas → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

