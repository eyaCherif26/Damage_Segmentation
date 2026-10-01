"""Merge per-image labelme-style annotation JSONs into one COCO file per split.

Generic over any labelme-style dataset (not hardcoded to dacl10k's location or class list) —
point it at any `--*_ann_dir`/`--*_img_dir` pair. Input schema expected per annotation file
(confirmed against dacl10k v2's `annotations/train/*.json`):

    {
      "imageWidth": int, "imageHeight": int, "imageName": str, "imagePath": str,
      "shapes": [{"label": str, "points": [[x, y], ...], "shape_type": "polygon"}, ...]
    }

Usage (both splits):
    python data/merge_dacl10k.py \\
        --train_ann_dir /path/to/annotations/train --train_img_dir /path/to/images/train \\
        --val_ann_dir   /path/to/annotations/validation --val_img_dir /path/to/images/val \\
        --output_dir data/processed/dacl10k_merged

Usage (a single homogeneous batch, no train/val split of its own — give only one pair; the other
is entirely optional):
    python data/merge_dacl10k.py \\
        --train_ann_dir /path/to/my_new_dataset/annotations --train_img_dir /path/to/my_new_dataset/images \\
        --output_dir data/processed/my_new_dataset_merged

Only `shape_type == "polygon"` is supported (matches the confirmed dacl10k v2 schema); any other
shape type raises rather than being silently skipped or mis-handled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _polygon_area_and_bbox(points: list[list[float]]) -> tuple[float, list[float]]:
    """Shoelace-formula area and COCO [x, y, w, h] bbox from a flat list of [x, y] points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2.0
    return area, [x_min, y_min, x_max - x_min, y_max - y_min]


def discover_categories(ann_dirs: list[Path]) -> list[str]:
    """Sorted unique label set across all given annotation directories — deterministic and
    dataset-agnostic (no hardcoded class list)."""
    labels: set[str] = set()
    for ann_dir in ann_dirs:
        for f in sorted(ann_dir.glob("*.json")):
            data = json.loads(f.read_text())
            for shape in data.get("shapes", []):
                labels.add(shape["label"])
    return sorted(labels)


def merge_split(
    ann_dir: Path,
    img_dir: Path,
    category_name_to_id: dict[str, int],
    split_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (coco_dict, report_dict) for one split."""
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_ann_id = 1
    zero_annotation_images = 0
    degenerate_dropped = 0
    per_class_counts: dict[str, int] = {name: 0 for name in category_name_to_id}

    ann_files = sorted(ann_dir.glob("*.json"))
    for ann_file in ann_files:
        data = json.loads(ann_file.read_text())
        image_name = data["imageName"]
        width, height = data["imageWidth"], data["imageHeight"]
        image_path = img_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(
                f"{ann_file} references image {image_name!r} but it does not exist under {img_dir}"
            )

        image_id = next_image_id
        next_image_id += 1
        images.append({"id": image_id, "file_name": image_name, "width": width, "height": height})

        n_valid_shapes = 0
        for shape in data.get("shapes", []):
            shape_type = shape.get("shape_type")
            if shape_type != "polygon":
                raise NotImplementedError(
                    f"{ann_file}: shape_type={shape_type!r} is not supported (only 'polygon' is "
                    "handled) — extend merge_split() deliberately if this dataset uses other "
                    "shape types, don't silently skip them."
                )
            points = shape["points"]
            if len(points) < 3:
                degenerate_dropped += 1
                continue
            area, bbox = _polygon_area_and_bbox(points)
            if area <= 0:
                degenerate_dropped += 1
                continue

            label = shape["label"]
            category_id = category_name_to_id[label]
            segmentation = [[coord for point in points for coord in point]]
            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1
            n_valid_shapes += 1
            per_class_counts[label] += 1

        if n_valid_shapes == 0:
            zero_annotation_images += 1

    categories = [{"id": cid, "name": name, "supercategory": "damage"} for name, cid in category_name_to_id.items()]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    report = {
        "split": split_name,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "zero_annotation_images": zero_annotation_images,
        "degenerate_polygons_dropped": degenerate_dropped,
        "per_class_instance_counts": per_class_counts,
    }
    return coco, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_ann_dir", type=Path, default=None)
    parser.add_argument("--train_img_dir", type=Path, default=None)
    parser.add_argument("--val_ann_dir", type=Path, default=None)
    parser.add_argument("--val_img_dir", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--categories_file",
        type=Path,
        default=None,
        help="Optional text file, one category name per line, fixing the id order. If omitted, "
        "categories are discovered from the data (sorted union of labels across every split given) "
        "— deterministic given fixed input data, but the order can shift if the input data changes.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    # Each split is independently optional (a dataset with only one homogeneous batch of labeled
    # images — no train/val distinction of its own — merges cleanly to just that one split), but a
    # split's ann_dir/img_dir must be given together, and at least one split must be given at all.
    splits: list[tuple[str, Path, Path]] = []
    for split, ann_dir, img_dir in (("train", args.train_ann_dir, args.train_img_dir), ("val", args.val_ann_dir, args.val_img_dir)):
        if (ann_dir is None) != (img_dir is None):
            raise SystemExit(f"--{split}_ann_dir and --{split}_img_dir must be given together")
        if ann_dir is not None:
            splits.append((split, ann_dir, img_dir))
    if not splits:
        raise SystemExit("give at least one of --train_ann_dir/--train_img_dir or --val_ann_dir/--val_img_dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, _, _ in splits:
        out_path = args.output_dir / split / "annotations.json"
        if out_path.exists() and not args.force:
            raise FileExistsError(f"{out_path} already exists. Pass --force to overwrite.")

    if args.categories_file:
        categories = [line.strip() for line in args.categories_file.read_text().splitlines() if line.strip()]
    else:
        categories = discover_categories([ann_dir for _, ann_dir, _ in splits])
    if not categories:
        raise ValueError("No categories discovered — check the given ann_dir(s) contain labelme JSON files.")
    category_name_to_id = {name: i + 1 for i, name in enumerate(categories)}  # 1-based: id 0 collides with background conventions

    reports = []
    for split, ann_dir, img_dir in splits:
        coco, report = merge_split(ann_dir, img_dir, category_name_to_id, split)
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        out_path = split_dir / "annotations.json"
        out_path.write_text(json.dumps(coco))
        print(f"[{split}] wrote {out_path}: {report['num_images']} images, {report['num_annotations']} annotations")
        reports.append(report)

    report_path = args.output_dir / "merge_report.json"
    report_path.write_text(json.dumps({"categories": category_name_to_id, "splits": reports}, indent=2))
    print(f"Wrote merge report to {report_path}")


if __name__ == "__main__":
    main()
