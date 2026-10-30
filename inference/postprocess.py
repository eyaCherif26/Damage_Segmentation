"""Threshold application and COCO-format export of sliding-window inference output."""
from __future__ import annotations

import json

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from skimage import measure

from metrics.instance_metrics import extract_instances, instance_full_mask


def apply_thresholds(prob_map: np.ndarray, class_names: list[str], thresholds: dict[str, float]) -> np.ndarray:
    """prob_map: (C, H, W). thresholds: {class_name: threshold}. Returns a (C, H, W) binary mask."""
    binary = np.zeros_like(prob_map, dtype=np.uint8)
    for i, name in enumerate(class_names):
        t = thresholds.get(name, 0.5)
        binary[i] = (prob_map[i] >= t).astype(np.uint8)
    return binary


def _polygon_rings(mask: np.ndarray) -> list[list[float]]:
    """mask: (H, W) bool, full image size, one instance. One flat [x1,y1,x2,y2,...] ring per
    contour found by skimage.measure.find_contours (marching squares at the 0.5 level -- hence the
    pixel-center '.5' coordinates), (row, col) swapped to (x, y). Directly drawable as an HTML
    <canvas> path or SVG <polygon> with no server-side RLE decoding, unlike this module's default
    "rle" segmentation format.

    `mask` is padded by 1px of background before tracing. Without this, an instance that touches the
    actual image border gets an OPEN (non-closed) contour there, which every consumer of
    `segmentation` draws as a closed polygon anyway -- implicitly connecting its two loose ends with
    a straight line that cuts across the real shape instead of following it (confirmed directly via
    icb-damage-detection/inference/predict.py's own identical algorithm, which hits this on every
    single instance since it crops to the tight bbox first -- see that file's own note). Padding is a
    no-op coordinate shift for the common case (an instance away from the image edge): pad +1,
    subtract 1 back out below."""
    padded = np.pad(mask.astype(bool), 1)
    rings = []
    for contour in measure.find_contours(padded.astype(np.float32), level=0.5):
        ring = []
        for r, c in contour:
            ring.append(float(c) - 1)
            ring.append(float(r) - 1)
        if len(ring) >= 6:  # at least 3 (x, y) points
            rings.append(ring)
    return rings


def _convex_hull(rings: list[list[float]]) -> list[list[float]]:
    """A single simplified convex-hull polygon (list of [x, y] pairs, not a flat ring like
    `segmentation`) over all of one instance's contour points combined -- a cheap, low-point-count
    outline for quick preview rendering alongside the full-detail `segmentation` rings."""
    points = np.array([[ring[i], ring[i + 1]] for ring in rings for i in range(0, len(ring), 2)], dtype=np.float32)
    if len(points) < 3:
        return [[float(x), float(y)] for x, y in points]
    hull = cv2.convexHull(points)
    return [[float(p[0][0]), float(p[0][1])] for p in hull]


def export_coco_predictions(
    binary_mask: np.ndarray,
    prob_map: np.ndarray,
    class_names: list[str],
    image_id: int,
    image_width: int,
    image_height: int,
    output_path: str,
    min_area: int = 16,
    file_name: str | None = None,
    segmentation_format: str = "rle",
) -> None:
    """binary_mask/prob_map: (C, H, W). Writes a COCO-format predictions file, one entry per
    connected component per class, scored by mean probability. Components smaller than `min_area`
    px are dropped before any mask is materialized; each kept instance is expanded to a full-size
    mask only long enough to encode it.

    `file_name`, if given, is recorded on the images[] entry -- previously omitted, but needed to
    know which image a prediction file actually belongs to (e.g. for visualization).

    `segmentation_format`: "rle" (default, byte-for-byte UNCHANGED from before this param existed --
    compact pycocotools RLE, needs a decoder to render) or "polygon" (one or more subpixel contour
    rings per instance via skimage.measure.find_contours, directly renderable in a browser with no
    decoding library, plus a simplified `hull` convex-hull field alongside the full-detail rings)."""
    if segmentation_format not in ("rle", "polygon"):
        raise ValueError(f"segmentation_format must be 'rle' or 'polygon', got {segmentation_format!r}")
    annotations = []
    next_ann_id = 1
    shape = (image_height, image_width)
    for class_idx, name in enumerate(class_names):
        instances = extract_instances(binary_mask[class_idx].astype(bool), prob_map[class_idx], min_area=min_area)
        for inst in instances:
            full = instance_full_mask(inst, shape)
            y0, x0, y1, x1 = inst["bbox"]
            ann = {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": class_idx + 1,
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
            }
            if segmentation_format == "rle":
                rle = mask_utils.encode(np.asfortranarray(full.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                ann["segmentation"] = rle
            else:
                rings = _polygon_rings(full)
                ann["hull"] = _convex_hull(rings) if rings else []
                ann["segmentation"] = rings
            # area/iscrowd/score last, matching icb-damage-detection/inference/predict.py's own
            # annotation field order exactly, and area as a float (not the plain int
            # extract_instances returns) to match that pipeline's own COCO export byte-for-byte.
            ann["area"] = float(inst["area"])
            ann["iscrowd"] = 0
            ann["score"] = inst["mean_prob"]
            annotations.append(ann)
            next_ann_id += 1

    image_entry = {"id": image_id}
    if file_name is not None:
        image_entry["file_name"] = file_name
    image_entry["width"] = image_width
    image_entry["height"] = image_height
    coco = {
        "images": [image_entry],
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
    }
    with open(output_path, "w") as f:
        json.dump(coco, f)
