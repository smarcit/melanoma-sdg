#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from PIL import Image

def collect_dermoscopic_bmps(root: Path):
    """
    Busca todos los BMP dentro de subdirectorios que terminan en 'Dermoscopic_Image'
    bajo cada caso (carpeta de imagen) directamente contenido en root.
    """
    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        # subcarpetas del caso; buscamos la que acabe en 'Dermoscopic_Image'
        derm_dirs = [d for d in case_dir.iterdir()
                     if d.is_dir() and d.name.endswith("Dermoscopic_Image")]
        if not derm_dirs:
            continue
        derm_dir = derm_dirs[0]
        bmps = list(derm_dir.glob("*.bmp")) + list(derm_dir.glob("*.BMP"))
        if not bmps:
            continue
        # En PH2 suele haber un único BMP
        yield case_dir.name, bmps[0]

def save_image(img: Image.Image, out_path: Path, fmt: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt.lower() == "png":
        img.save(out_path, format="PNG", optimize=True)
    elif fmt.lower() in ("jpg", "jpeg"):
        # calidad alta y sin subsampling para minimizar artefactos
        img = img.convert("RGB")  # asegurar 3 canales
        img.save(out_path, format="JPEG", quality=95, subsampling=0, optimize=True)
    else:
        raise ValueError(f"Formato no soportado: {fmt}")

def main():
    parser = argparse.ArgumentParser(
        description="Extrae y convierte las imágenes dermatoscópicas de PH2 a un directorio 'images'.")
    parser.add_argument("--input-root", required=True,
                        help="Ruta al directorio 'PH2 Dataset images'")
    parser.add_argument("--output-dir", required=True,
                        help="Ruta al directorio de salida (p.ej. .../PH2/images)")
    parser.add_argument("--format", default="png", choices=["png", "jpg"],
                        help="Formato de salida (png por defecto)")
    parser.add_argument("--force", action="store_true",
                        help="Sobrescribir si el archivo de salida ya existe")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    fmt = args.format.lower()

    if not input_root.exists():
        print(f"[ERROR] No existe: {input_root}", file=sys.stderr)
        sys.exit(1)

    total = 0
    skipped = 0
    written = 0
    missing = 0

    for case_name, bmp_path in collect_dermoscopic_bmps(input_root):
        total += 1
        # Nombre de salida = nombre base del BMP (o del caso) con nueva extensión
        # PH2 suele usar IMDxxx.bmp; usamos el stem del BMP
        stem = bmp_path.stem if bmp_path.stem else case_name
        out_path = output_dir / f"{stem}.{fmt}"

        if out_path.exists() and not args.force:
            skipped += 1
            print(f"[SKIP] Existe: {out_path}")
            continue

        try:
            with Image.open(bmp_path) as im:
                # Convertir a RGB si fuese paleta/grayscale para consistencia
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                save_image(im, out_path, fmt)
                written += 1
                print(f"[OK] {bmp_path} -> {out_path}")
        except Exception as e:
            missing += 1
            print(f"[ERROR] {bmp_path}: {e}", file=sys.stderr)

    print("\nResumen:")
    print(f"  Casos encontrados:    {total}")
    print(f"  Guardados/convertidos:{written}")
    print(f"  Ya existían (skip):   {skipped}")
    print(f"  Errores:              {missing}")

if __name__ == "__main__":
    main()

