#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import yaml

def to_binary(mapped_class_8: str, binary_rules: dict) -> str:
    if not mapped_class_8 or mapped_class_8.strip() == "":
        return ""
    if mapped_class_8 == binary_rules.get("melanoma", "melanoma"):
        return "melanoma"
    return binary_rules.get("default_non_melanoma", "no_melanoma")

def main():
    ap = argparse.ArgumentParser(description="Build taxonomy_map.csv from taxonomy_map.yaml")
    ap.add_argument("--config", required=True, help="Path to taxonomy_map.yaml")
    ap.add_argument("--out", required=True, help="Output CSV path (taxonomy_map.csv)")
    ap.add_argument("--source-ref", default="", help="Default source reference (overrides YAML if given)")
    ap.add_argument("--version", default="", help="Override version")
    ap.add_argument("--default-rationale", default="yaml_rule_v1",
                    help="Default rationale for mapped entries (pending entries use 'pending_decision')")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    datasets = cfg.get("datasets", {}) or {}
    datasets_unmapped = cfg.get("datasets_unmapped", {}) or {}
    binary_rules = cfg.get("binary_rules", {}) or {}
    version = args.version or cfg.get("version", "1.0")
    default_rationale = args.default_rationale
    default_source_ref = args.source_ref or cfg.get("source_ref", "")

    rows = []

    # 1) datasets con mapeo definido
    for dataset, mapping in datasets.items():
        if mapping is None:
            continue
        for original_label, mapped_class_8 in mapping.items():
            mapped_class_8 = mapped_class_8 or ""
            mapped_binary = to_binary(mapped_class_8, binary_rules) if mapped_class_8 else ""
            rows.append({
                "dataset": dataset,
                "original_label": original_label,
                "mapped_class_8": mapped_class_8,
                "mapped_binary": mapped_binary,
                "subclass_detail": "",
                "mapping_rationale": (default_rationale if mapped_class_8 else "pending_decision"),
                "source_ref": default_source_ref,
                "version": version
            })

    # 2) datasets_unmapped: etiquetas pendientes (sin mapeo aún)
    for dataset, labels in datasets_unmapped.items():
        for original_label in (labels or []):
            rows.append({
                "dataset": dataset,
                "original_label": original_label,
                "mapped_class_8": "",
                "mapped_binary": "",
                "subclass_detail": "",
                "mapping_rationale": "pending_decision",
                "source_ref": default_source_ref,
                "version": version
            })

    # escribir CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset","original_label","mapped_class_8","mapped_binary",
            "subclass_detail","mapping_rationale","source_ref","version"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] taxonomy_map.csv -> {out_path} ({len(rows)} rows)")

if __name__ == "__main__":
    main()

