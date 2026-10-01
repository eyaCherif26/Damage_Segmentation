"""Does dacl10k's carved-val (10% of train, used for per-epoch monitoring/threshold selection)
look like dacl10k's official val (promoted to test, per CLAUDE.md's user-confirmed strategy)?

Read-only: loads the persisted carve and the two merged COCO files, maps categories to the 5 MVP
classes with the same label map training uses, and reports per-class pixel/instance share, image
count, and resolution for each pool. A large mismatch here means part of any val->test Dice drop
is distribution shift, not (only) overfitting.

    python scripts/eda_dacl_carve_vs_test.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pycocotools import mask as mask_utils

from data.label_mapping import LabelMapping

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP_CLASSES = ("Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence")


def _polygon_area(seg: list, height: int, width: int) -> float:
    rles = mask_utils.frPyObjects([p for p in seg if len(p) >= 6], height, width)
    return float(mask_utils.area(mask_utils.merge(rles))) if rles else 0.0


def _pool_stats(coco_path: Path, image_ids: set[int], mapping: LabelMapping, cat_id_to_name: dict[int, str]) -> dict:
    with open(coco_path) as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"] if img["id"] in image_ids}

    pixel_area = defaultdict(float)
    instance_count = defaultdict(int)
    for ann in coco["annotations"]:
        if ann["image_id"] not in images:
            continue
        mvp_class = mapping.resolve(cat_id_to_name[ann["category_id"]])
        if mvp_class is None:
            continue
        seg = ann["segmentation"]
        img = images[ann["image_id"]]
        area = _polygon_area(seg, img["height"], img["width"]) if isinstance(seg, list) else float(mask_utils.area(seg))
        pixel_area[mvp_class] += area
        instance_count[mvp_class] += 1

    widths = [img["width"] for img in images.values()]
    heights = [img["height"] for img in images.values()]
    total_image_pixels = sum(w * h for w, h in zip(widths, heights)) or 1.0
    return {
        "n_images": len(images),
        "mean_resolution": (sum(widths) / len(widths), sum(heights) / len(heights)) if widths else (0, 0),
        "pixel_share": {c: pixel_area.get(c, 0.0) / total_image_pixels for c in MVP_CLASSES},
        "instance_count": {c: instance_count.get(c, 0) for c in MVP_CLASSES},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--carve", type=Path, default=REPO_ROOT / "data/splits/dacl10k_train_val_carve_seed42.json")
    p.add_argument("--train_coco", type=Path, default=REPO_ROOT / "data/processed/dacl10k_merged/train/annotations.json")
    p.add_argument("--test_coco", type=Path, default=REPO_ROOT / "data/processed/dacl10k_merged/val/annotations.json")
    p.add_argument("--label_map", type=Path, default=REPO_ROOT / "configs/data/label_maps/dacl10k_19_to_mvp5.yaml")
    p.add_argument("--output", type=Path, default=None, help="optional path to also write the report as JSON")
    args = p.parse_args()

    with open(args.carve) as f:
        carve = json.load(f)
    carved_val_ids = set(carve["val_ids"])
    with open(args.train_coco) as f:
        train_categories = {c["id"]: c["name"] for c in json.load(f)["categories"]}
    with open(args.test_coco) as f:
        test_coco_raw = json.load(f)
    test_ids = {img["id"] for img in test_coco_raw["images"]}
    mapping = LabelMapping.load(args.label_map)

    carved_val = _pool_stats(args.train_coco, carved_val_ids, mapping, train_categories)
    official_test = _pool_stats(args.test_coco, test_ids, mapping, {c["id"]: c["name"] for c in test_coco_raw["categories"]})

    print(f"carved-val (10% of dacl10k train, seed in {args.carve.name}): {carved_val['n_images']} images, "
          f"mean resolution {carved_val['mean_resolution'][0]:.0f}x{carved_val['mean_resolution'][1]:.0f}")
    print(f"official test (dacl10k official validation split):        {official_test['n_images']} images, "
          f"mean resolution {official_test['mean_resolution'][0]:.0f}x{official_test['mean_resolution'][1]:.0f}")
    print(f"\n{'class':<15} {'val pixel%':>12} {'test pixel%':>12} {'ratio':>8}   {'val #inst':>10} {'test #inst':>10}")
    for c in MVP_CLASSES:
        v, t = carved_val["pixel_share"][c] * 100, official_test["pixel_share"][c] * 100
        ratio = (v / t) if t > 0 else float("inf") if v > 0 else float("nan")
        print(f"{c:<15} {v:>11.3f}% {t:>11.3f}% {ratio:>7.2f}x   {carved_val['instance_count'][c]:>10} {official_test['instance_count'][c]:>10}")

    report = {"carved_val": carved_val, "official_test": official_test}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
