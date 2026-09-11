#!/usr/bin/env python3
import argparse, csv, os, re, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm

# ---------- utils ----------
def _strip_accents(s: str) -> str:
    try:
        return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    except Exception:
        return s

def _norm(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    s = _strip_accents(s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).lower()
    return s

def safe_open_image(path: Path):
    try:
        with Image.open(path) as im:
            im.load()
            return im.copy()
    except Exception:
        return None

def image_info(path: Path):
    im = safe_open_image(path)
    if im is None:
        return None, None, None, False
    w, h = im.size
    mode = im.mode
    channels = 1 if mode in ("1","L") else (4 if "A" in mode else 3)
    is_rgb = (mode in ("RGB","RGBA"))
    return w, h, channels, is_rgb

def compute_phash(path: Path):
    try:
        with Image.open(path) as im:
            return str(imagehash.phash(im))
    except Exception:
        return ""

def load_taxonomy_map(taxonomy_csv: Path):
    """ Espera: dataset, original_label, superclass, binary (1/0) """
    df = pd.read_csv(taxonomy_csv)
    mapping = {}
    for _, r in df.iterrows():
        d  = _norm(r.get("dataset",""))
        ol = _norm(r.get("original_label",""))
        sc = "" if pd.isna(r.get("superclass","")) else str(r.get("superclass","")).strip()
        binv = r.get("binary","")
        try: binv = "" if pd.isna(binv) else str(int(binv))
        except Exception: binv = str(binv).strip()
        if d and ol:
            mapping[(d, ol)] = (sc, binv)
    return mapping

def map_via_taxonomy(mapping, dataset, original_label):
    return mapping.get((_norm(dataset), _norm(original_label)), ("", ""))

def parse_types(arg: str):
    if not arg: return {}
    out = {}
    for kv in arg.split(","):
        if ":" in kv:
            k, v = kv.split(":", 1)
            out[k.strip()] = v.strip()
    return out

def _get(row, name, default=""):
    try: return row.get(name, default)
    except Exception: return default

# ---------- dataset iterators (siguen tu guía) ----------
def iter_derm12345(root: Path):
    """
    DERM12345:
      isic_id -> image_id
      diagnosis_3 -> original_label   (taxonomy_map hará superclass/binary)
      diagnosis_4 -> subclass
      anatom_site_general -> anatom_site_general
      copyright -> license
      image_type -> acquisition_type
    """
    # 1) Cargar metadata
    meta = None
    for fname in ["DERM12345_metadata.csv", "metadata.csv"]:
        mp = root / fname
        if mp.exists():
            meta = pd.read_csv(mp)
            break
    if meta is None:
        raise SystemExit("[DERM12345] metadata CSV not found in: " + str(root))

    # 2) Normalizar nombres de columnas
    def _n(s):
        return str(s).strip().lower().replace("-", " ").replace("_", " ").replace("  ", " ")
    meta.rename(columns={c: _n(c) for c in meta.columns}, inplace=True)

    # 3) Índice por isic_id
    if "isic id" in meta.columns:
        meta.set_index("isic id", inplace=True)
    elif "isic_id" in meta.columns:
        meta.set_index("isic_id", inplace=True)
    else:
        raise SystemExit("[DERM12345] expected column 'isic_id' / 'ISIC ID'")

    # 4) Directorio de imágenes (¡aquí estaba el NameError!)
    images_dir = root / "images"
    if not images_dir.exists():
        raise SystemExit("[DERM12345] images dir not found: " + str(images_dir))

    # 5) Iterar archivos
    for p in images_dir.rglob("*.*"):
        if not p.is_file():
            continue
        image_id = p.stem

        original_label = ""
        subclass = ""
        anatom = ""
        license_s = ""
        acq_type = ""

        if image_id in meta.index:
            row = meta.loc[image_id]
            # helpers seguros para .get en Series
            def g(r, key, alt=""):
                try:
                    return r.get(key, alt)
                except Exception:
                    return alt

            original_label = str(g(row, "diagnosis 3", g(row, "diagnosis_3", "")))
            subclass       = str(g(row, "diagnosis 4", g(row, "diagnosis_4", "")))
            anatom         = g(row, "anatom site general", g(row, "anatom_site_general", ""))
            license_s      = g(row, "copyright", "")
            acq_type       = g(row, "image type", g(row, "image_type", ""))

        yield {
            "dataset": "DERM12345",
            "image_id": image_id,
            "filepath": str(p),

            # taxonomy_map hará el mapeo → dejamos superclass vacío
            "original_label": original_label,
            "superclass": "",
            "subclass": subclass,

            "age": "", "sex": "", "fitzpatrick": "",
            "anatom_site_general": anatom,
            "diagnosis_method": "",
            "acquisition_type": acq_type,

            "license": license_s,
            "source_url_or_doi": ""
        }

def iter_ham10000(root: Path):
    """
    HAM10000:
      image_id -> image_id
      dx -> original_label (abreviado: akiec/bcc/bkl/df/nv/mel/vasc)
      dx_type -> diagnosis_method
      age -> age
      sex -> sex
      localization -> anatom_site_general
    """
    # 1) Cargar metadata
    meta = None
    for fname in ["HAM10000_metadata.csv", "metadata.csv"]:
        mp = root / fname
        if mp.exists():
            meta = pd.read_csv(mp)
            break
    if meta is not None:
        idx = "image_id" if "image_id" in meta.columns else ("image" if "image" in meta.columns else None)
        if idx:
            meta = meta.set_index(idx)

    # 2) Directorio de imágenes
    images_dir = root / "images"
    if not images_dir.exists():
        raise SystemExit("[HAM10000] images dir not found: " + str(images_dir))

    # 3) Iterar archivos
    for p in images_dir.rglob("*.*"):
        if not p.is_file():
            continue
        image_id = p.stem

        original_label = ""  # dx abreviado
        age = sex = anatom = dx_type = ""

        if meta is not None and image_id in meta.index:
            row = meta.loc[image_id]
            original_label = str(_get(row, "dx", ""))          # <- ABREVIADO (clave taxonomy)
            age            = _get(row, "age", "")
            sex            = _get(row, "sex", "")
            anatom         = _get(row, "localization", "")
            dx_type        = _get(row, "dx_type", "")

        yield {
            "dataset": "HAM10000",
            "image_id": image_id,
            "filepath": str(p),

            # ¡no fijamos superclass aquí! lo pondrá taxonomy_map
            "original_label": original_label,
            "superclass": "",
            "subclass": "",

            "age": age,
            "sex": sex,
            "anatom_site_general": anatom,
            "diagnosis_method": dx_type,

            "fitzpatrick": "",
            "acquisition_type": "dermoscopic",
            "license": "",
            "source_url_or_doi": ""
        }

def iter_isic2019(root: Path, taxonomy: dict):
    """
    ISIC2019:
      one-hot -> original_label abreviado (MEL/NV/BCC/AK/BKL/DF/VASC/SCC)
      superclass y binary las fijará taxonomy_map.csv
    """
    # 1) Cargar metadata one-hot
    meta_path = root / "ISIC2019_metadata.csv"
    if not meta_path.exists():
        raise SystemExit("[ISIC2019] metadata.csv (one-hot) not found in: " + str(root))
    meta = pd.read_csv(meta_path)
    if "image" in meta.columns:
        meta = meta.set_index("image")

    # 2) Directorio de imágenes
    images_dir = root / "images"
    if not images_dir.exists():
        raise SystemExit("[ISIC2019] images dir not found: " + str(images_dir))

    # 3) Iterar archivos
    for p in images_dir.rglob("*.*"):
        if not p.is_file():
            continue
        image_id = p.stem

        original_label = ""  # abreviado
        if image_id in meta.index:
            row = meta.loc[image_id]
            for col in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]:
                if col in row.index and str(row[col]).strip() in ("1", "1.0", "True", "true"):
                    original_label = col
                    break

        yield {
            "dataset": "ISIC2019",
            "image_id": image_id,
            "filepath": str(p),

            # ¡no fijamos superclass aquí! lo pondrá taxonomy_map
            "original_label": original_label,
            "superclass": "",
            "subclass": "",

            "age": "",
            "sex": "",
            "anatom_site_general": "",
            "diagnosis_method": "",
            "fitzpatrick": "",
            "acquisition_type": "dermoscopic",
            "license": "",
            "source_url_or_doi": ""
        }

def iter_ph2(root: Path):
    """
    PH2: cabecera desplazada + clínica en 3 columnas + colores multi-columna.
    Detecta automáticamente la fila de cabecera (buscando 'Image Name'/'Image').
    """
    import numpy as np

    # --- 1) Localizar el archivo Excel ---
    xpaths = [root/"PH2_metadata.xlsx", root/"PH2_metadata.xls", root/"meta.xlsx"]
    xpath = None
    for xp in xpaths:
        if xp.exists():
            xpath = xp; break
    if xpath is None:
        raise SystemExit("[PH2] No se ha encontrado PH2_metadata.xlsx/.xls")

    # --- 2) Leer sin cabecera y detectar la fila header ---
    raw = pd.read_excel(xpath, header=None)  # no header todavía
    def norm(s):
        return (
            str(s).strip().lower()
            .replace("-", " ").replace("_", " ")
            .replace("  ", " ")
        )
    # candidatos de nombre de columna:
    wanted = {"image name", "image"}

    header_row = None
    # escanear primeras ~50 filas por si hay leyendas al principio
    scan_upto = min(60, len(raw))
    for r in range(scan_upto):
        row_vals = [norm(x) for x in raw.iloc[r].tolist()]
        if any(v in wanted for v in row_vals):
            header_row = r
            break
    if header_row is None:
        raise SystemExit("[PH2] No encuentro la fila de cabecera (buscando 'Image Name'/'Image').")

    # --- 3) Reconstruir el DataFrame con esa fila como cabecera ---
    cols = raw.iloc[header_row].tolist()
    meta = raw.iloc[header_row+1:].copy()
    meta.columns = cols
    # eliminar columnas 100% vacías y filas sin imagen
    meta = meta.dropna(how="all", axis=1)
    meta = meta.dropna(how="all", axis=0)

    # normalizar nombres de columnas
    meta = meta.rename(columns={c: norm(c) for c in meta.columns})

    # helpers para encontrar columnas por alias (soporta abreviaturas/truncados)
    def pick(*aliases):
        aliases = [norm(a) for a in aliases]
        # exact match
        for a in aliases:
            if a in meta.columns: return a
        # fuzzy (contiene/prefijo/sufijo)
        for a in aliases:
            for c in meta.columns:
                if c.startswith(a) or c.endswith(a) or a in c:
                    return c
        return None

    col_img   = pick("image name", "image")
    if not col_img:
        raise SystemExit("[PH2] No encuentro columna de nombre de imagen ('Image Name'/'Image').")

    col_histo = pick("histological diagnosis")
    col_asym  = pick("assymetry", "asymmetry")

    # clínica en 3 columnas
    col_common   = pick("common nevus")
    col_atypical = pick("atypical nevus")
    col_melanoma = pick("melanoma")
    if not (col_common and col_atypical and col_melanoma):
        raise SystemExit("[PH2] Faltan columnas clínicas (Common/Atypical/Melanoma).")

    # rasgos dermatoscópicos (texto A/T/AT, A/P…)
    col_pigment  = pick("pigment network")
    col_dots     = pick("dots globules", "dots", "globules")
    col_streaks  = pick("streaks")
    col_regress  = pick("regression areas")
    col_blueveil = pick("blue whitish veil", "blue whitish", "blue whitish veil (a/p)")

    # colores (marcados con 'X')
    col_white    = pick("white")
    col_red      = pick("red")
    col_lbrown   = pick("light brown", "light brow")
    col_dbrown   = pick("dark brown", "dark brow")
    col_bgray    = pick("blue gray", "blue gra")
    col_black    = pick("black")

    # set índice por nombre de imagen
    meta[col_img] = meta[col_img].astype(str).str.strip()
    meta = meta.set_index(col_img)

    # --- 4) Recorremos imágenes en /images ---
    images_dir = root / "images"
    for p in images_dir.rglob("*.*"):
        if not p.is_file():
            continue
        image_id = p.stem

        def is_marked(val):
            s = str(val).strip().lower()
            return (s == "x") or (s == "1") or (s == "true")

        clinical = ""; histo = ""; asym = ""
        pigment = dots = streaks = regress = blueveil = ""
        colors = dict(white=0, red=0, light_brown=0, dark_brown=0, blue_gray=0, black=0)

        if image_id in meta.index:
            row = meta.loc[image_id]

            # Clinical Diagnosis (prioridad: Melanoma > Atypical > Common)
            if is_marked(row.get(col_melanoma, "")):
                clinical = "Melanoma"
            elif is_marked(row.get(col_atypical, "")):
                clinical = "Atypical Nevus"
            elif is_marked(row.get(col_common, "")):
                clinical = "Common Nevus"
            else:
                clinical = ""

            if col_histo: histo = str(row.get(col_histo, "")).strip()
            if col_asym:  asym  = str(row.get(col_asym, "")).strip()

            if col_pigment:  pigment  = str(row.get(col_pigment, "")).strip().upper()
            if col_dots:     dots     = str(row.get(col_dots, "")).strip().upper()
            if col_streaks:  streaks  = str(row.get(col_streaks, "")).strip().upper()
            if col_regress:  regress  = str(row.get(col_regress, "")).strip().upper()
            if col_blueveil: blueveil = str(row.get(col_blueveil, "")).strip().upper()

            def mk(cname):
                v = row.get(cname, "") if cname else ""
                return 1 if is_marked(v) else 0
            colors = {
                "white":       mk(col_white),
                "red":         mk(col_red),
                "light_brown": mk(col_lbrown),
                "dark_brown":  mk(col_dbrown),
                "blue_gray":   mk(col_bgray),
                "black":       mk(col_black),
            }

        yield {
            "dataset":"PH2",
            "image_id":image_id,
            "filepath":str(p),

            # clave para taxonomy_map (PH2): usar clínica como original_label
            "original_label": clinical,
            "superclass":"", "subclass":"",

            "clinical_diagnosis": clinical,
            "histological_diagnosis": histo,
            "asymmetry": asym,
            "pigment_network": pigment,
            "dots_globules": dots,
            "streaks": streaks,
            "regression_areas": regress,
            "blue_whitish_veil": blueveil,

            "color_white": colors["white"],
            "color_red": colors["red"],
            "color_light_brown": colors["light_brown"],
            "color_dark_brown": colors["dark_brown"],
            "color_blue_gray": colors["blue_gray"],
            "color_black": colors["black"],

            "age":"","sex":"","fitzpatrick":"",
            "anatom_site_general":"", "diagnosis_method":"",
            "acquisition_type":"dermoscopic",
            "license":"", "source_url_or_doi":""
        }

# ---------- worker ----------
def process_row(row, skip_phash: bool, override_types: dict):
    p = Path(row["filepath"])
    w = h = channels = None; is_rgb = False; is_corrupt = False; ph = ""
    info = image_info(p)
    if info == (None, None, None, False):
        is_corrupt = True
    else:
        w, h, channels, is_rgb = info
        if not skip_phash:
            ph = compute_phash(p)

    acq = row.get("acquisition_type","")
    if row["dataset"] in override_types: acq = override_types[row["dataset"]]

    # clave global única del proyecto (evita colisiones de image_id)
    global_id = f'{row["dataset"]}:{row["image_id"]}'

    out = {
        "global_id": global_id,
        "image_id": row["image_id"],
        "dataset": row["dataset"],
        "filepath": str(p),
        "split": "train",
        "width": w, "height": h, "channels": channels, "is_rgb": is_rgb,
        "original_label": row.get("original_label",""),
        "superclass": row.get("superclass",""),
        "subclass": row.get("subclass",""),
        "binary": "",  # opcional: lo podemos llenar al cruzar con taxonomy_map si quieres
        "age": row.get("age",""), "sex": row.get("sex",""), "fitzpatrick": row.get("fitzpatrick",""),
        "anatom_site_general": row.get("anatom_site_general",""),
        "acquisition_type": acq,
        "diagnosis_method": row.get("diagnosis_method",""),
        "clinical_diagnosis": row.get("clinical_diagnosis",""),
        "dermoscopic_diagnosis": row.get("dermoscopic_diagnosis",""),
        "histological_diagnosis": row.get("histological_diagnosis",""),
        "asymmetry": row.get("asymmetry",""),
        "color_white": row.get("color_white",""),
        "color_red": row.get("color_red",""),
        "color_light_brown": row.get("color_light_brown",""),
        "color_dark_brown": row.get("color_dark_brown",""),
        "color_blue_gray": row.get("color_blue_gray",""),
        "color_black": row.get("color_black",""),
        "is_corrupt": is_corrupt, "phash": ph, "dup_group_id": "", "quality_flags": "",
        "hair_removed": False, "padding_applied": False, "norm_scheme": "",
        "license": row.get("license",""), "source_url_or_doi": "",
        "is_synthetic": False, "synth_method": "", "config_id": "", "seed": "", "parent_real_id": ""
    }
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Build master_metadata (v3) con mapping exacto por dataset")
    ap.add_argument("--taxonomy", required=False, help="taxonomy_map.csv (opcional, para validar/cargar binary)")
    ap.add_argument("--out", required=True, help="master_metadata.csv")
    ap.add_argument("--derm12345", help="raíz DERM12345")
    ap.add_argument("--ham", help="raíz HAM10000")
    ap.add_argument("--isic2019", help="raíz ISIC2019")
    ap.add_argument("--ph2", help="raíz PH2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-phash", action="store_true")
    ap.add_argument("--dataset-types", default="", help='Overrides: "HAM10000:dermoscopic,PH2:dermoscopic"')
    ap.add_argument("--log-every", type=int, default=2000)
    args = ap.parse_args()

    taxonomy = load_taxonomy_map(Path(args.taxonomy)) if args.taxonomy else {}
    override_types = parse_types(args.dataset_types)

    raw = []
    if args.derm12345: raw += list(iter_derm12345(Path(args.derm12345)))
    if args.ham:       raw += list(iter_ham10000(Path(args.ham)))
    if args.isic2019:  raw += list(iter_isic2019(Path(args.isic2019), taxonomy))
    if args.ph2:       raw += list(iter_ph2(Path(args.ph2)))

    # resumen entrada
    by_ds = {}
    for r in raw: by_ds[r["dataset"]] = by_ds.get(r["dataset"],0)+1
    print("Resumen entrada:")
    for ds,n in by_ds.items(): print(f"  {ds}: {n}")

    # procesar
    res = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_row, r, args.skip_phash, override_types) for r in raw]
        pbar = tqdm(total=len(futs), desc="Extrayendo metadatos")
        done=0
        for f in as_completed(futs):
            res.append(f.result()); done += 1; pbar.update(1)
            if args.log_every and done % args.log_every == 0:
                print(f"[LOG] {done}/{len(futs)}")
        pbar.close()

    # si hay taxonomy_map: rellenar 'binary' donde aplique y verificar coherencia de 'superclass'
    if taxonomy:
        unmapped = {}

        def fill_from_taxonomy(row):
            sc_t, bin_t = map_via_taxonomy(taxonomy, row["dataset"], row["original_label"])
            if sc_t:
                # imponemos taxonomy como fuente de verdad
                if row.get("superclass") and row["superclass"] != sc_t:
                    row["quality_flags"] = (row.get("quality_flags", "") + "|superclass_mismatch").strip("|")
                row["superclass"] = sc_t
            else:
                # contar no mapeados por dataset/label
                key = (row["dataset"], row["original_label"])
                unmapped[key] = unmapped.get(key, 0) + 1
            if bin_t and not row.get("binary"):
                row["binary"] = bin_t
            return row

        res = [fill_from_taxonomy(r) for r in res]

        # Log útil: top-10 labels sin mapeo
        misses = sorted(unmapped.items(), key=lambda x: -x[1])[:10]
        if misses:
            print("\n[WARN] Etiquetas sin mapeo en taxonomy_map (top-10):")
            for (ds, lab), cnt in misses:
                print(f"  {ds} :: '{lab}' -> {cnt} filas")

    # DataFrame + columnas finales
    cols = [
        "global_id","image_id","dataset","filepath","split","width","height","channels","is_rgb",
        "original_label","superclass","subclass","binary",
        "age","sex","fitzpatrick","anatom_site_general","acquisition_type","diagnosis_method",
        "clinical_diagnosis","dermoscopic_diagnosis","histological_diagnosis","asymmetry",
        "color_white","color_red","color_light_brown","color_dark_brown","color_blue_gray","color_black",
        "is_corrupt","phash","dup_group_id","quality_flags",
        "hair_removed","padding_applied","norm_scheme",
        "license","source_url_or_doi",
        "is_synthetic","synth_method","config_id","seed","parent_real_id"
    ]
    df = pd.DataFrame(res, columns=cols)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\n[OK] master_metadata v3 -> {outp} ({len(df)} filas)")

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS","1")
    main()
