"""Threshold application and COCO-format export of sliding-window inference output."""
from __future__ import annotations

import json

import numpy as np
from pycocotools import mask as mask_utils

from metrics.instance_metrics import extract_instances, instance_full_mask


def apply_thresholds(prob_map: np.ndarray, class_names: list[str], thresholds: dict[str, float]) -> np.ndarray:
    """prob_map: (C, H, W). thresholds: {class_name: threshold}. Returns a (C, H, W) binary mask."""
    binary = np.zeros_like(prob_map, dtype=np.uint8)
    for i, name in enumerate(class_names):
        t = thresholds.get(name, 0.5)
        binary[i] = (prob_map[i] >= t).astype(np.uint8)
    return binary


def export_coco_predictions(
    binary_mask: np.ndarray,
    prob_map: np.ndarray,
    class_names: list[str],
    image_id: int,
    image_width: int,
    image_height: int,
    output_path: str,
    min_area: int = 16,
) -> None:
    """binary_mask/prob_map: (C, H, W). Writes a COCO-format predictions file (RLE segmentation,
    one entry per connected component per class, scored by mean probability). Components smaller
    than `min_area` px are dropped before any mask is materialized; each kept instance is
    expanded to a full-size mask only long enough to RLE-encode it."""
    annotations = []
    next_ann_id = 1
    shape = (image_height, image_width)
    for class_idx, name in enumerate(class_names):
        instances = extract_instances(binary_mask[class_idx].astype(bool), prob_map[class_idx], min_area=min_area)
        for inst in instances:
            full = instance_full_mask(inst, shape)
            rle = mask_utils.encode(np.asfortranarray(full.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            y0, x0, y1, x1 = inst["bbox"]
            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": class_idx + 1,
                    "segmentation": rle,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "area": inst["area"],
                    "score": inst["mean_prob"],
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1

    coco = {
        "images": [{"id": image_id, "width": image_width, "height": image_height}],
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
    }
    with open(output_path, "w") as f:
        json.dump(coco, f)
