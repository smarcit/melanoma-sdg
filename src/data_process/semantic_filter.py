# semantic_filter_synth_timm_evalstyle.py
# -*- coding: utf-8 -*-
import argparse, csv
from pathlib import Path
from typing import List, Tuple

import torch, timm
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -------------------------- Dataset simple por paths --------------------------
class ImagePathsDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            im = Image.open(p).convert("RGB")
            x = self.transform(im)
            return x, str(p)
        except (UnidentifiedImageError, OSError, ValueError):
            # devolvemos tensor dummy si falla, el caller filtrará
            return None, str(p)


# -------------------------- Args --------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Semantic filter (melanoma) for synthetic images using a timm model (eval-style)."
    )
    ap.add_argument("--syn-dir", type=str, required=True,
                    help="Directorio con imágenes sintéticas (ya filtradas por RGB/blur/duplicados).")
    ap.add_argument("--arch", type=str, default="tf_efficientnetv2_s.in1k",
                    help="Nombre del modelo timm (ej: tf_efficientnetv2_s.in1k).")
    ap.add_argument("--num-classes", type=int, default=8,
                    help="Nº de clases de la cabeza tal y como se entrenó.")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Ruta al checkpoint .pt/.pth con el state_dict del modelo.")
    ap.add_argument("--img-size", type=int, default=384,
                    help="Tamaño de entrada (coincidir con validación/entrenamiento).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--use-amp", action="store_true", help="Activar AMP como en los evals.")
    ap.add_argument("--mel-idx", type=int, default=4,
                    help="Índice de la clase 'melanoma' en la salida (multiclase).")
    ap.add_argument("--thresh", type=float, default=0.65,
                    help="Umbral T* de aceptación (prob. mínima de melanoma).")
    ap.add_argument("--out-accept", type=str, default="./synthetic_melanoma_accepted.csv")
    ap.add_argument("--out-reject", type=str, default="./synthetic_melanoma_rejected.csv")
    ap.add_argument("--out-scores", type=str, default="./synthetic_melanoma_scores.csv",
                    help="CSV con p_melanoma para todas las imágenes (debug/opcional).")
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


# -------------------------- Modelo & transforms (eval-style) --------------------------
def build_model_timm(arch: str, num_classes: int, ckpt_path: str, device: str):
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # limpia prefijos típicos de DDP o envoltorios
    state = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {list(missing)[:6]} ...")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {list(unexpected)[:6]} ...")
    model.to(device).eval()
    return model

def build_transform(img_size: int):
    # Igual que en eval_effnet.py: Resize -> CenterCrop -> ToTensor -> Normalize ImageNet
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])


# -------------------------- Utilidades --------------------------
def list_images(root: Path) -> List[Path]:
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def write_csv(path: Path, rows: List[Tuple[str, float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "p_melanoma"])
        w.writeheader()
        for pth, pm in rows:
            w.writerow({"path": pth, "p_melanoma": pm})

@torch.no_grad()
def infer_probs(model, dl, device: str, mel_idx: int, use_amp: bool):
    results = []  # [(path, p_melanoma)]
    # contexto AMP igual que en los evals
    if use_amp:
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.startswith("cuda") else torch.autocast("cpu")
    else:
        class _NoOp:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        amp_ctx = _NoOp()

    with amp_ctx:
        for batch in tqdm(dl, desc="Semantic filter", leave=False):
            x, paths = batch
            # filtra entradas None (fallo al abrir)
            keep = [i for i, t in enumerate(x) if t is not None] if isinstance(x, list) else None
            if isinstance(x, list):
                if not keep:  # todo el batch falló
                    for p in paths: results.append((p, -1.0))
                    continue
                x = torch.stack([x[i] for i in keep], dim=0)
                paths = [paths[i] for i in keep]
            x = x.to(device, non_blocking=True)

            logits = model(x)  # [B, C]
            if logits.ndim != 2:
                raise RuntimeError(f"Salida inesperada del modelo: shape={tuple(logits.shape)}")
            C = logits.shape[1]
            if C == 1:
                probs = torch.sigmoid(logits[:, 0])
            elif C == 2:
                probs = torch.softmax(logits, dim=1)[:, mel_idx]
            else:
                probs = torch.softmax(logits, dim=1)[:, mel_idx]
            probs = probs.float().cpu().numpy().tolist()
            results.extend(list(zip(paths, probs)))
    return results


# -------------------------- Main --------------------------
def main():
    args = parse_args()
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    model = build_model_timm(args.arch, args.num_classes, args.ckpt, device)
    tfm = build_transform(args.img_size)

    syn_dir = Path(args.syn_dir)
    files = list_images(syn_dir)
    if not files:
        raise SystemExit(f"No se encontraron imágenes en {syn_dir}")

    ds = ImagePathsDataset(files, tfm)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.startswith("cuda"))
    )

    results = infer_probs(model, dl, device, args.mel_idx, args.use_amp)

    accepted = [(p, pm) for (p, pm) in results if pm >= args.thresh]
    rejected = [(p, pm) for (p, pm) in results if 0.0 <= pm < args.thresh]  # pm<0 fueron fallos de lectura

    write_csv(Path(args.out_accept), accepted)
    write_csv(Path(args.out_reject), rejected)

    # CSV con todas las puntuaciones (útil para auditoría/curva precisión-recall si te hace falta)
    Path(args.out_scores).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_scores, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path","p_melanoma"])
        w.writeheader()
        for p, pm in results:
            w.writerow({"path": p, "p_melanoma": pm})

    tot = len(results)
    keep = len(accepted)
    print(f"[semantic] aceptadas: {keep}/{tot} ({100.0*keep/max(1,tot):.1f}%) con T*={args.thresh}")
    print(f"[out] accept → {args.out_accept}")
    print(f"[out] reject → {args.out_reject}")
    print(f"[out] scores → {args.out_scores}")


if __name__ == "__main__":
    main()
