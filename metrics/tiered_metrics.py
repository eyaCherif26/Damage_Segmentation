"""Tiered evaluation — Tier 1 (Detection) / Tier 2 (Localization) / Tier 3 (Extent). See
CLAUDE.md's 2026-10-02 status entry and scripts/evaluate.py --tiered_report for the full
rationale. Pure pixel/area segmentation metrics (no connected-component/instance logic anywhere
in this module, by deliberate scope decision — see CLAUDE.md).

Purpose: strict pixel Dice/IoU (metrics/pixel_metrics.py) conflates "didn't find the damage" with
"found it, drew it a few pixels off" — both read as a low score. Each tier relaxes one dimension
of strictness so those two failure modes can be told apart:
  Tier 1 (is it there at all?): image- and tile-level presence/absence, ignoring shape/position.
  Tier 2 (is it roughly in the right place?): buffered precision/recall within a tolerance radius,
    ignoring exact extent.
  Tier 3 (how good are the outlines/sizes?): today's pooled Dice/IoU (reused, not recomputed) plus
    area-error and thin-class skeleton width.

Reuses rather than reimplements: metrics.presence_metrics.PresenceMetricAccumulator (Tier 1, both
image- and tile-level — the same accumulator, instantiated twice, fed whole images vs. tile-sized
slices), metrics.boundary_metrics.skeleton_length_and_width (Tier 3 thin-class width),
metrics.pixel_metrics.PixelMetricAccumulator's existing result (Tier 3 Dice/IoU — read directly
from the result every eval run already produces, never recomputed here).

All tolerance/threshold choices (buffer_px, min_gt_px, min_pred_px, tile_px) are FIXED CHOICES
passed in from scripts/evaluate.py's --tier_* flags, set BEFORE looking at results — this module
never tunes or defaults any of them to "whatever looks good" (buffer_px/min_gt_px/min_pred_px have
no CLI default at all, forcing an explicit, deliberate choice).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from metrics.boundary_metrics import skeleton_length_and_width
from metrics.presence_metrics import PresenceMetricAccumulator


def _iter_tiles(arr: np.ndarray, tile_px: int):
    """arr: (C, H, W). Yields non-overlapping tile_px x tile_px (C, h, w) slices in row-major
    order. Edge tiles may be smaller than tile_px x tile_px (image not evenly divisible) — scored
    as-is over however many real pixels they contain, never padded or dropped."""
    _, h, w = arr.shape
    for y0 in range(0, h, tile_px):
        for x0 in range(0, w, tile_px):
            yield arr[:, y0:y0 + tile_px, x0:x0 + tile_px]


def _distance_near_mask(binary: np.ndarray, buffer_px: float) -> np.ndarray:
    """(H, W) bool -> (H, W) bool: True where a pixel is within `buffer_px` Euclidean distance of a
    foreground pixel of `binary`, via scipy.ndimage.distance_transform_edt — a genuine circular
    distance transform, distinct from metrics.boundary_metrics.boundary_f1's existing tolerance
    mechanism (a square binary_dilation structuring element): the two will disagree slightly at the
    same nominal pixel radius, don't treat `--tier_buffer_px N` and boundary_f1's `tolerance_px N`
    as the same number. All-background input -> all-False (no 'near' zone around nothing)."""
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    return distance_transform_edt(~binary.astype(bool)) <= buffer_px


@dataclass
class TieredAccumulator:
    """Fed from the SAME already-thresholded, validity-masked (pred_bin, gt_bin) (C, H, W) bool
    arrays every other accumulator in this pipeline receives (metrics/evaluation.py's
    _PerImageMetrics/PresenceMetricAccumulator) — no extra inference, no new data path."""

    class_names: list[str]
    thin_classes: list[str]
    buffer_px: float
    min_gt_px: int
    min_pred_px: int
    tile_px: int = 128
    tile_min_gt_px: int | None = None
    tile_min_pred_px: int | None = None

    def __post_init__(self) -> None:
        self._thin = set(self.thin_classes)
        self._image_presence = PresenceMetricAccumulator(self.class_names, self.min_gt_px, self.min_pred_px)
        self._tile_presence = PresenceMetricAccumulator(
            self.class_names,
            self.tile_min_gt_px if self.tile_min_gt_px is not None else self.min_gt_px,
            self.tile_min_pred_px if self.tile_min_pred_px is not None else self.min_pred_px,
        )
        n = len(self.class_names)
        # Tier 2: pooled buffered precision/recall — raw pixel counts summed dataset-wide first,
        # ratio computed once in compute() (same aggregate-then-ratio discipline as
        # metrics.pixel_metrics.PixelMetricAccumulator's tp/fp/fn).
        self._pred_total = np.zeros(n, dtype=np.int64)
        self._pred_in_gt_near = np.zeros(n, dtype=np.int64)
        self._gt_total = np.zeros(n, dtype=np.int64)
        self._gt_in_pred_near = np.zeros(n, dtype=np.int64)
        # Tier 3: area-error and thin-class width are inherently PER-IMAGE ratios, so kept as raw
        # per-image lists and aggregated as a distribution (median/p90/mean) in compute() — the
        # same discipline metrics.evaluation.py::_PerImageMetrics already uses for its own
        # boundary_f1/cldice per-image score lists.
        self._area_error: dict[str, list[float]] = defaultdict(list)
        self._pred_area_sum = np.zeros(n, dtype=np.int64)
        self._gt_area_sum = np.zeros(n, dtype=np.int64)
        self._gt_width: dict[str, list[float]] = defaultdict(list)
        self._pred_width: dict[str, list[float]] = defaultdict(list)

    def update(self, pred_bin: np.ndarray, gt_bin: np.ndarray) -> None:
        """pred_bin/gt_bin: (C, H, W) bool, already thresholded and validity-masked."""
        self._image_presence.update(pred_bin, gt_bin)
        for pred_tile, gt_tile in zip(_iter_tiles(pred_bin, self.tile_px), _iter_tiles(gt_bin, self.tile_px)):
            self._tile_presence.update(pred_tile, gt_tile)

        for c, name in enumerate(self.class_names):
            p, g = pred_bin[c], gt_bin[c]
            if not p.any() and not g.any():
                continue  # matches _PerImageMetrics' own convention for boundary_f1/cldice/etc.

            gt_near = _distance_near_mask(g, self.buffer_px)
            pred_near = _distance_near_mask(p, self.buffer_px)
            self._pred_total[c] += int(p.sum())
            self._pred_in_gt_near[c] += int((p & gt_near).sum())
            self._gt_total[c] += int(g.sum())
            self._gt_in_pred_near[c] += int((g & pred_near).sum())

            pred_area, gt_area = int(p.sum()), int(g.sum())
            self._pred_area_sum[c] += pred_area
            self._gt_area_sum[c] += gt_area
            if gt_area > 0:
                self._area_error[name].append(abs(pred_area - gt_area) / gt_area)

            if name in self._thin:
                _, gt_w = skeleton_length_and_width(g)
                _, pred_w = skeleton_length_and_width(p)
                if not np.isnan(gt_w):
                    self._gt_width[name].append(gt_w)
                if not np.isnan(pred_w):
                    self._pred_width[name].append(pred_w)

    def compute(self) -> dict[str, Any]:
        """Point estimates only (no confidence interval — see CLAUDE.md for why bootstrap CIs were
        scoped out of this pass: no site/asset metadata exists anywhere in this codebase today, so
        a resample could only be per-image, reflecting frame-to-frame noise, not the site-to-site
        variation the idea was meant to bound)."""
        tier2 = {}
        tier3 = {}
        for c, name in enumerate(self.class_names):
            tier2[name] = {
                "buffered_precision": float(self._pred_in_gt_near[c] / self._pred_total[c]) if self._pred_total[c] > 0 else float("nan"),
                "buffered_recall": float(self._gt_in_pred_near[c] / self._gt_total[c]) if self._gt_total[c] > 0 else float("nan"),
            }
            errs = self._area_error[name]
            entry = {
                "area_error_median": float(np.median(errs)) if errs else float("nan"),
                "area_error_p90": float(np.percentile(errs, 90)) if errs else float("nan"),
                "total_area_ratio": float(self._pred_area_sum[c] / self._gt_area_sum[c]) if self._gt_area_sum[c] > 0 else float("nan"),
            }
            if name in self._thin:
                gt_w, pred_w = self._gt_width[name], self._pred_width[name]
                entry["gt_mean_width_px"] = float(np.mean(gt_w)) if gt_w else float("nan")
                entry["pred_mean_width_px"] = float(np.mean(pred_w)) if pred_w else float("nan")
            tier3[name] = entry
        return {
            "tier1_image": self._image_presence.compute(),
            "tier1_tile": self._tile_presence.compute(),
            "tier2": tier2,
            "tier3": tier3,
        }


def tiered_results_to_dataframe(tiered: TieredAccumulator, pixel_overall: dict, class_names: list[str], split: str) -> pd.DataFrame:
    """Assembles the Tier 1/2/3 report table as a SEPARATE file (results_<name>_tiered.csv), never
    merged into results_<name>.csv — nothing already parsing that file's columns is affected.
    Tier 3's dice/iou are joined verbatim from `pixel_overall` (= evaluate_split's/
    evaluate_external_with_tiled_inference's existing result["overall"] dict) rather than
    recomputed here — a direct test asserts this equals the SAME number already in
    results_<name>.csv for the same class."""
    result = tiered.compute()
    t1i, t1t = result["tier1_image"], result["tier1_tile"]
    rows = []
    for name in class_names:
        i, t = t1i["per_class"][name], t1t["per_class"][name]
        m2, m3 = result["tier2"][name], result["tier3"][name]
        pix = pixel_overall["per_class"][name]
        row = {
            "split": split, "class": name,
            "image_recall": i["recall_presence"], "image_precision": i["precision_presence"], "image_f1": i["f1_presence"],
            "image_n_with_gt": i["n_images_with_gt"], "image_n_flagged": i["n_images_flagged"],
            "tile_recall": t["recall_presence"], "tile_precision": t["precision_presence"], "tile_f1": t["f1_presence"],
            "tile_n_with_gt": t["n_images_with_gt"], "tile_n_flagged": t["n_images_flagged"],
            "buffered_precision": m2["buffered_precision"], "buffered_recall": m2["buffered_recall"],
            "dice": pix["dice"], "iou": pix["iou"],
            "area_error_median": m3["area_error_median"], "area_error_p90": m3["area_error_p90"],
            "total_area_ratio": m3["total_area_ratio"],
            "gt_mean_width_px": m3.get("gt_mean_width_px", float("nan")),
            "pred_mean_width_px": m3.get("pred_mean_width_px", float("nan")),
        }
        rows.append(row)
    rows.append({"split": split, "class": "__summary_image__", "image_n_with_gt": t1i["n_images"],
                 "exact_match_ratio": t1i["exact_match_ratio"], "hamming_accuracy": t1i["hamming_accuracy"]})
    rows.append({"split": split, "class": "__summary_tile__", "tile_n_with_gt": t1t["n_images"],
                 "exact_match_ratio": t1t["exact_match_ratio"], "hamming_accuracy": t1t["hamming_accuracy"]})
    return pd.DataFrame(rows)
