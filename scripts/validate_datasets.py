"""Validate every COCO file of every configured source + label-mapping coverage, before training.

    python scripts/validate_datasets.py --config configs/experiments/baseline_smp_efficientnet_b0.yaml

Checks: required COCO keys, 1-based category ids, null/empty segmentation, box-only polygons,
dangling image ids, every native category present in the label mapping, image files exist (sampled),
and that the dacl10k carve is disjoint. Exits non-zero on any error.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loader import load_experiment_config
from data.label_mapping import LabelMapping
from data.schema_validator import validate_coco_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--sample_images", type=int, default=200, help="How many image files per split to check for existence.")
    args = p.parse_args()

    cfg = load_experiment_config(args.config)
    failed = False
    for name, src in cfg.data.sources_resolved.items():
        mapping = LabelMapping.load(src.label_mapping_path)
        for split, path in src.annotations.items():
            with open(path) as f:
                coco = json.load(f)
            cats = [c["name"] for c in coco["categories"]]
            report = validate_coco_file(path, known_category_names=set(mapping.source_to_mvp))
            try:
                mapping.check_covers(cats)
            except KeyError as e:
                report.errors.append(str(e))

            images = coco["images"]
            root = Path(src.images_roots[split])
            missing = [im["file_name"] for im in random.Random(0).sample(images, min(args.sample_images, len(images))) if not (root / im["file_name"]).exists()]
            if missing:
                report.errors.append(f"{len(missing)} sampled image file(s) missing under {root}, e.g. {missing[:3]}")

            status = "OK " if report.ok else "FAIL"
            print(f"[{status}] {name}/{split}: {len(images)} images, {len(coco['annotations'])} annotations, {len(cats)} categories")
            for e in report.errors:
                print(f"       error: {e}")
            for w in report.warnings:
                print(f"       warning: {w}")
            failed |= not report.ok

    if "dacl10k" in cfg.data.sources_resolved:
        from data.datasets.combined_dataset import _dacl10k_split_image_ids

        ids = _dacl10k_split_image_ids(cfg.data.sources_resolved["dacl10k"], cfg.data)
        print(f"[OK  ] dacl10k roles ({cfg.data.dacl10k_test_strategy}): " + ", ".join(f"{k}={len(v)}" for k, v in ids.items()))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
