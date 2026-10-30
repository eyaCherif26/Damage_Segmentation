"""Image-level ("did it notice the damage") presence/absence metrics — a coarse, client-facing
complement to this project's strict pixel-level (metrics/pixel_metrics.py), boundary/topology
(metrics/boundary_metrics.py), and instance-shape (metrics/instance_metrics.py) metrics.

Per class, per image: does the ground truth contain any pixel of this class ANYWHERE in the image
(ignoring exact shape/extent/position), and did the prediction flag it too. This answers "did the
model correctly tell us what kinds of damage are in this photo" rather than "how precisely did it
draw the boundary" — a genuinely different, much more lenient question, deliberately reported
alongside the strict metrics rather than instead of them (see scripts/evaluate.py's
--presence_report, default off, additive-only: it never changes any existing output file).

Exact Match Ratio (EMR) here is the same metric as a reference `torchmetrics`-style
all-classes-correct-or-nothing accumulator, just derived from dense segmentation masks via an
any() reduction over (H, W) instead of from a native per-sample multi-label classification vector.
"""
from __future__ import annotations

import numpy as np


class PresenceMetricAccumulator:
    """Fed from the SAME already-thresholded, validity-masked (pred_bin, gt_bin) arrays
    metrics/evaluation.py::_PerImageMetrics.update() already receives — no extra inference, no
    extra cost, just an additional reduction of data already in hand."""

    def __init__(self, class_names: list[str], min_gt_px: int = 1, min_pred_px: int = 1) -> None:
        """`min_gt_px`/`min_pred_px` (both default 1, i.e. today's exact `.any()`-based behavior,
        unchanged): a class only counts as "present" if its total area is at least this many
        pixels — a noise-filtering floor for scripts/evaluate.py --tiered_report's Tier 1 detection
        test, where a single stray pixel (annotation speckle or a one-pixel false positive) shouldn't
        count as "found"/"flagged" the same way a real region of damage does."""
        self.class_names = class_names
        self.min_gt_px = min_gt_px
        self.min_pred_px = min_pred_px
        n = len(class_names)
        self.tp = np.zeros(n, dtype=np.int64)
        self.fp = np.zeros(n, dtype=np.int64)
        self.fn = np.zeros(n, dtype=np.int64)
        self.tn = np.zeros(n, dtype=np.int64)
        self.exact_matches = 0
        self.hamming_correct = 0
        self.n_images = 0

    def update(self, pred_bin: np.ndarray, gt_bin: np.ndarray) -> None:
        """pred_bin/gt_bin: (C, H, W) bool, already thresholded and validity-masked."""
        if self.min_gt_px <= 1 and self.min_pred_px <= 1:
            pred_present = pred_bin.any(axis=(1, 2))  # exact original code path, unchanged
            gt_present = gt_bin.any(axis=(1, 2))
        else:
            gt_present = gt_bin.sum(axis=(1, 2)) >= self.min_gt_px
            pred_present = pred_bin.sum(axis=(1, 2)) >= self.min_pred_px
        self.tp += pred_present & gt_present
        self.fp += pred_present & ~gt_present
        self.fn += ~pred_present & gt_present
        self.tn += ~pred_present & ~gt_present
        self.exact_matches += int(np.all(pred_present == gt_present))
        self.hamming_correct += int((pred_present == gt_present).sum())
        self.n_images += 1

    def compute(self) -> dict:
        """Per class: NaN (not 0) when there's nothing to compute from — e.g. recall_presence is
        NaN if this class never appears in the ground truth at all — matching
        metrics/pixel_metrics.py's "no GT pixels -> NaN, exclude from macro_present" convention,
        rather than silently reading 0."""
        per_class = {}
        for i, name in enumerate(self.class_names):
            n_with_gt = int(self.tp[i] + self.fn[i])
            n_flagged = int(self.tp[i] + self.fp[i])
            recall = float(self.tp[i] / n_with_gt) if n_with_gt > 0 else float("nan")
            precision = float(self.tp[i] / n_flagged) if n_flagged > 0 else float("nan")
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else float("nan")
            per_class[name] = {
                "n_images_with_gt": n_with_gt,
                "n_images_flagged": n_flagged,
                "tp": int(self.tp[i]), "fp": int(self.fp[i]), "fn": int(self.fn[i]), "tn": int(self.tn[i]),
                "recall_presence": recall,
                "precision_presence": precision,
                "f1_presence": f1,
            }
        return {
            "per_class": per_class,
            # exact_match_ratio: fraction of images where EVERY class's presence/absence matched —
            # the strict, all-or-nothing headline number (see EMR in the module docstring).
            "exact_match_ratio": self.exact_matches / self.n_images if self.n_images else float("nan"),
            # hamming_accuracy: mean per-(image,class) correctness — a softer middle ground.
            "hamming_accuracy": self.hamming_correct / (self.n_images * len(self.class_names)) if self.n_images else float("nan"),
            "n_images": self.n_images,
        }
