# quality_metrics_min_progress.py
import os, shutil, tempfile, random, sys, threading, time
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import torch
from torch_fidelity import calculate_metrics

# --- CONFIG ---
REAL_DIR = Path("../../data/imagefolder/test/melanoma")     # ISIC2019 (no visto por el generador)
SYN_DIR  = Path("../../data/gen/SD_E2_HOPE/v1_8000")    # tus sintéticos filtrados
OUT_PNG  = Path("../figures/synthetic_quality_metrics.png")
USE_LPIPS = True          # pon False si quieres aún más mínimo
SAMPLE_SIZE = None        # por ej. 5000 para probar rápido; None = todo

SPINNER_INTERVAL = 2.0    # segundos entre ticks del spinner


def materialize_rgb_only(src: Path) -> Path:
    import numpy as np, cv2
    tmp = Path(tempfile.mkdtemp(prefix=f"qm_{src.name}_"))
    ok = 0
    files = [p for p in src.rglob("*") if p.is_file()]
    print(f"[prep] {src} → {len(files)} ficheros. Validando/normalizando a RGB…")
    for p in files:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                (tmp / p.name).parent.mkdir(parents=True, exist_ok=True)
                im.save(tmp / p.name, format="PNG")
                ok += 1
        except (UnidentifiedImageError, OSError):
            continue
    if ok == 0:
        raise RuntimeError(f"No se pudo materializar ninguna imagen válida desde {src}")
    print(f"[prep] OK: {ok} imágenes válidas.")
    return tmp


def spinner(msg: str, stop_event: threading.Event):
    symbols = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{msg} {symbols[i%len(symbols)]}")
        sys.stdout.flush()
        i += 1
        time.sleep(SPINNER_INTERVAL)
    sys.stdout.write("\r" + " "*(len(msg)+2) + "\r")
    sys.stdout.flush()


def main():
    # 1) Normaliza a RGB (evita fallos del DataLoader)
    real_tmp = materialize_rgb_only(REAL_DIR)
    synth_tmp = materialize_rgb_only(SYN_DIR)

    # 2) Lanza cálculo con monitor de progreso sencillo
    print("[fid/kid] Extrayendo features y calculando métricas… (esto puede tardar)")
    stop = threading.Event()
    t = threading.Thread(target=spinner, args=("  trabajando", stop))
    t.daemon = True
    t.start()

    try:
        metrics = calculate_metrics(
            input1=str(real_tmp),
            input2=str(synth_tmp),
            cuda=torch.cuda.is_available(),
            fid=True,
            kid=True,
            isc=False,
            prc=False,
            verbose=True,                 # logs de torch-fidelity
            dataloader_num_workers=0,     # evita errores de worker
            batch_size=32,
            sample_size=SAMPLE_SIZE       # limita nº de imágenes opcionalmente
        )
    finally:
        stop.set()
        t.join()

    fid_val = float(metrics["frechet_inception_distance"])
    kid_mean = float(metrics["kernel_inception_distance_mean"])
    kid_std  = float(metrics["kernel_inception_distance_std"])

    # 3) (Opcional) LPIPS rápido intra-sintético con progreso simple
    lpips_mean = None
    if USE_LPIPS:
        print("[lpips] Calculando LPIPS en una muestra…")
        import lpips, numpy as np
        model = lpips.LPIPS(net='alex').eval()
        synth_files = [p for p in Path(synth_tmp).iterdir() if p.is_file()]
        random.seed(42); random.shuffle(synth_files)
        synth_files = synth_files[:40]
        scores = []
        for i in range(len(synth_files)-1):
            im1 = Image.open(synth_files[i]).convert("RGB")
            im2 = Image.open(synth_files[i+1]).convert("RGB")
            t1 = lpips.im2tensor(np.array(im1)[:, :, ::-1])
            t2 = lpips.im2tensor(np.array(im2)[:, :, ::-1])
            with torch.no_grad():
                s = model(t1, t2).item()
            scores.append(s)
            if (i+1) % 5 == 0:
                print(f"  pares LPIPS: {i+1}/{len(synth_files)-1}")
        lpips_mean = sum(scores)/len(scores)

    # 4) Plot mínimo
    import matplotlib.pyplot as plt
    labels = ["FID", "KID(mean)"]
    vals   = [fid_val, kid_mean]
    if lpips_mean is not None:
        labels.append("LPIPS(mean)")
        vals.append(lpips_mean)

    plt.figure(figsize=(5,3.2))
    plt.bar(labels, vals)
    plt.ylabel("valor (menor es mejor para FID/KID)")
    plt.title("Calidad imágenes sintéticas de melanoma")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=220)

    print(f"\nResultados → FID={fid_val:.3f} | KID={kid_mean:.3f}±{kid_std:.3f}" + (f" | LPIPS={lpips_mean:.3f}" if lpips_mean is not None else ""))
    print(f"Figura guardada en: {OUT_PNG}")


if __name__ == "__main__":
    main()

