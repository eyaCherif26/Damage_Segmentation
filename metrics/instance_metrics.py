"""Instance-level metrics bridged from dense per-class probability maps via connected components.

Default: connected-component-based, size-bucketed detection rate (not full COCO mask AP) — see
CLAUDE.md/docs/methodology.md for why: the model only outputs a dense probability map (no native
instance separation), and dacl10k/cif_coco GT polygons of the same class can genuinely overlap, so
strict AP's one-prediction-per-GT matching would penalize a correct CC-merged prediction. Optional
`pycocotools` mask AP remains available as a secondary number (config-gated) for literature
comparability.

Memory: instances are stored as bbox-cropped masks, never full-image masks. A noisy prediction on
a large orthophoto can contain 10^5 components; one full-size mask per component would need
hundreds of GB.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import label as cc_label
from skimage.measure import regionprops

SIZE_BUCKETS = {"small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, float("inf"))}


@dataclass
class InstanceDetectionResult:
    size_bucket_recall: dict[str, float]
    num_gt_instances: dict[str, int]
    num_detected: dict[str, int]


def extract_instances(
    binary_mask: np.ndarray,
    prob_map: np.ndarray | None = None,
    min_area: int = 1,
) -> list[dict]:
    """binary_mask: (H, W) thresholded prediction or GT for one class. Returns one dict per
    8-connected component with area >= min_area:
    {"bbox": (y0, x0, y1, x1), "crop": bool array of the bbox size, "area": int, "mean_prob": float}.
    Use `instance_full_mask` to materialize a full-size mask for a single instance on demand."""
    labeled = cc_label(binary_mask, connectivity=2)
    instances = []
    for region in regionprops(labeled):
        if region.area < min_area:
            continue
        y0, x0, y1, x1 = region.bbox
        crop = region.image
        if prob_map is not None:
            mean_prob = float(prob_map[y0:y1, x0:x1][crop].mean())
        else:
            mean_prob = 1.0
        instances.append({"bbox": (y0, x0, y1, x1), "crop": crop, "area": int(region.area), "mean_prob": mean_prob})
    return instances


def instance_full_mask(instance: dict, shape: tuple[int, int]) -> np.ndarray:
    """Materializes one instance as a full-size (H, W) bool mask. Allocates H*W bytes — call per
    instance and discard, never keep a list of these."""
    full = np.zeros(shape, dtype=bool)
    y0, x0, y1, x1 = instance["bbox"]
    full[y0:y1, x0:x1] = instance["crop"]
    return full


def _bucket_for_area(area: float) -> str:
    for name, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= area < hi:
            return name
    return "large"


def size_bucketed_detection_rate(
    pred_binary: np.ndarray,
    target_binary: np.ndarray,
    iou_threshold: float = 0.5,
) -> InstanceDetectionResult:
    """For each GT instance (connected component of `target_binary`), it counts as "detected" if
    ANY predicted connected component overlaps it at IoU >= `iou_threshold`. Unlike AP's strict
    one-prediction-per-GT matching, one predicted component may be credited to several GT
    instances. Caveat: IoU is computed per GT instance, so a prediction that merges GT blobs into
    a region much larger than each of them is NOT credited (IoU falls below the threshold) —
    the metric tolerates modest merging, not arbitrary merging.

    Works directly on the two label images (bincount over the pixels under each GT component),
    so cost is proportional to GT area, independent of how many predicted components exist."""
    gt_labeled = cc_label(target_binary, connectivity=2)
    pred_labeled = cc_label(pred_binary, connectivity=2)
    pred_areas = np.bincount(pred_labeled.ravel())

    bucket_hits = {b: 0 for b in SIZE_BUCKETS}
    bucket_totals = {b: 0 for b in SIZE_BUCKETS}

    for region in regionprops(gt_labeled):
        bucket = _bucket_for_area(region.area)
        bucket_totals[bucket] += 1

        y0, x0, y1, x1 = region.bbox
        pred_under_gt = pred_labeled[y0:y1, x0:x1][region.image]
        overlap_counts = np.bincount(pred_under_gt)
        detected = False
        for pred_label in np.nonzero(overlap_counts)[0]:
            if pred_label == 0:
                continue
            intersection = overlap_counts[pred_label]
            union = region.area + pred_areas[pred_label] - intersection
            if union > 0 and intersection / union >= iou_threshold:
                detected = True
                break
        if detected:
            bucket_hits[bucket] += 1

    recall = {b: (bucket_hits[b] / bucket_totals[b] if bucket_totals[b] > 0 else float("nan")) for b in SIZE_BUCKETS}
    return InstanceDetectionResult(
        size_bucket_recall=recall,
        num_gt_instances=bucket_totals,
        num_detected=bucket_hits,
    )
