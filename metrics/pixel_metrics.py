"""Per-class + macro/frequency-weighted pixel metrics (IoU, Dice, precision, recall), accumulated
across a whole split so results are exact (not a mean-of-per-batch-means), plus a per-class
threshold sweep for the "threshold selection per class on the validation set" requirement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class PixelMetricAccumulator:
    class_names: list[str]
    threshold: float | torch.Tensor = 0.5
    tp: torch.Tensor = field(init=False)
    fp: torch.Tensor = field(init=False)
    fn: torch.Tensor = field(init=False)
    gt_pixels: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.class_names)
        self.tp = torch.zeros(n, dtype=torch.float64)
        self.fp = torch.zeros(n, dtype=torch.float64)
        self.fn = torch.zeros(n, dtype=torch.float64)
        self.gt_pixels = torch.zeros(n, dtype=torch.float64)

    def update(self, probs: torch.Tensor, target: torch.Tensor, validity_mask: torch.Tensor | None = None) -> None:
        """probs/target: (B, C, H, W). validity_mask: (B, 1, H, W) or None."""
        threshold = self.threshold
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.view(1, -1, 1, 1)
        preds = (probs >= threshold).float()
        if validity_mask is not None:
            preds = preds * validity_mask
            target = target * validity_mask
        dims = (0, 2, 3)
        self.tp += (preds * target).sum(dims).double().cpu()
        self.fp += (preds * (1 - target)).sum(dims).double().cpu()
        self.fn += ((1 - preds) * target).sum(dims).double().cpu()
        self.gt_pixels += target.sum(dims).double().cpu()

    def compute(self, eps: float = 1e-6) -> dict:
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)

        weights = self.gt_pixels / self.gt_pixels.sum().clamp_min(eps)
        present = self.gt_pixels > 0

        per_class = {
            name: {
                "iou": iou[i].item(),
                "dice": dice[i].item(),
                "precision": precision[i].item(),
                "recall": recall[i].item(),
                "gt_pixels": int(self.gt_pixels[i].item()),
            }
            for i, name in enumerate(self.class_names)
        }
        return {
            "per_class": per_class,
            # mean over classes that actually occur in the ground truth: a dataset that lacks a class must not
            # have that class's meaningless 0 drag the average down (`macro` keeps all classes, as trained on)
            "macro_present": {
                k: (v[present].mean().item() if present.any() else float("nan"))
                for k, v in (("iou", iou), ("dice", dice), ("precision", precision), ("recall", recall))
            },
            "macro": {
                "iou": iou.mean().item(),
                "dice": dice.mean().item(),
                "precision": precision.mean().item(),
                "recall": recall.mean().item(),
            },
            "frequency_weighted": {
                "iou": (iou * weights).sum().item(),
                "dice": (dice * weights).sum().item(),
                "precision": (precision * weights).sum().item(),
                "recall": (recall * weights).sum().item(),
            },
        }


class ThresholdSweepAccumulator:
    """Streaming per-class threshold sweep: accumulates TP/FP/FN at every candidate threshold
    batch by batch (on whatever device the tensors live on), so choosing thresholds on a large val
    set never requires holding all probability maps in memory. Only ever feed it val data."""

    def __init__(self, class_names: list[str], thresholds: np.ndarray | None = None) -> None:
        self.class_names = class_names
        self.thresholds = np.arange(0.05, 0.951, 0.05) if thresholds is None else np.asarray(thresholds)
        k, c = len(self.thresholds), len(class_names)
        self.tp = torch.zeros(k, c, dtype=torch.float64)
        self.fp = torch.zeros(k, c, dtype=torch.float64)
        self.fn = torch.zeros(k, c, dtype=torch.float64)
        self.gt_pixels = torch.zeros(c, dtype=torch.float64)

    def update(self, probs: torch.Tensor, target: torch.Tensor, validity_mask: torch.Tensor | None = None) -> None:
        if validity_mask is not None:
            target = target * validity_mask
        dims = (0, 2, 3)
        self.gt_pixels += target.sum(dims).double().cpu()
        for i, t in enumerate(self.thresholds):
            preds = (probs >= float(t)).float()
            if validity_mask is not None:
                preds = preds * validity_mask
            self.tp[i] += (preds * target).sum(dims).double().cpu()
            self.fp[i] += (preds * (1 - target)).sum(dims).double().cpu()
            self.fn[i] += ((1 - preds) * target).sum(dims).double().cpu()

    def dice_table(self, eps: float = 1e-6) -> np.ndarray:
        """(num_thresholds, num_classes) Dice at each candidate threshold."""
        return (2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)).numpy()

    def best_thresholds(self, default: float = 0.5) -> dict[str, dict[str, float]]:
        """Per class: the Dice-maximizing threshold and its Dice. Classes with no GT pixels in the
        sweep data can't be tuned, so they fall back to `default` (and report NaN Dice)."""
        table = self.dice_table()
        out = {}
        for c, name in enumerate(self.class_names):
            if self.gt_pixels[c] == 0:
                out[name] = {"threshold": default, "val_dice": float("nan")}
                continue
            k = int(np.argmax(table[:, c]))
            out[name] = {"threshold": float(self.thresholds[k]), "val_dice": float(table[k, c])}
        return out


def sweep_per_class_thresholds(
    probs: torch.Tensor,
    target: torch.Tensor,
    class_names: list[str],
    validity_mask: torch.Tensor | None = None,
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """probs/target: (N, C, H, W) held fully in memory for one val pass (call on a val-sized
    accumulation, not the full train set). Returns {class_name: best_threshold}, each chosen
    independently to maximize that class's Dice — the "threshold selection per class on the
    validation set" requirement. Must only ever be called against val, never test."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    best_threshold = {}
    for i, name in enumerate(class_names):
        best_dice, best_t = -1.0, 0.5
        for t in thresholds:
            acc = PixelMetricAccumulator([name], threshold=float(t))
            acc.update(probs[:, i : i + 1], target[:, i : i + 1], validity_mask)
            dice = acc.compute()["per_class"][name]["dice"]
            if dice > best_dice:
                best_dice, best_t = dice, float(t)
        best_threshold[name] = best_t
    return best_threshold
