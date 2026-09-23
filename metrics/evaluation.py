"""Full evaluation protocol (see PROMPT.md "Evaluation"):

1. Per-class thresholds are chosen on the VAL split only (Dice-maximizing, streamed on the GPU).
2. Those fixed thresholds are applied to the TEST split (never tuned on test).
3. Reported per class and overall (macro + frequency-weighted): pixel IoU / Dice / precision /
   recall; boundary F1 for every class and clDice for thin-structure classes; instance-level
   detection rate by size bucket. Everything is computed inside the validity mask.

Per-image metrics (boundary F1, clDice, instance detection) skip an image for a class when that
class is absent from BOTH ground truth and prediction — counting those as a perfect 1.0 would
inflate the score on rare classes. Instance size buckets are defined in pixels at evaluation
resolution (`data.image_size`), not native resolution / mm.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metrics.boundary_metrics import boundary_f1, hard_cl_dice
from metrics.instance_metrics import SIZE_BUCKETS, size_bucketed_detection_rate
from metrics.pixel_metrics import PixelMetricAccumulator, ThresholdSweepAccumulator


def load_thresholds(path: str | Path, class_names: list[str]) -> dict[str, float] | None:
    """Reads the `thresholds.json` written by scripts/evaluate.py (`{class: {"threshold": t, "val_dice": d}}`).
    Returns None if the file does not exist; raises if it exists but does not cover every class, since a
    silently partial threshold file would give misleading predictions."""
    path = Path(path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    missing = [c for c in class_names if c not in raw]
    if missing:
        raise KeyError(f"{path} has no threshold for class(es) {missing}; it covers {sorted(raw)}. "
                       "Re-run scripts/evaluate.py with this experiment's class list.")
    return {c: float(raw[c]["threshold"] if isinstance(raw[c], dict) else raw[c]) for c in class_names}


@torch.no_grad()
def _predict(model: torch.nn.Module, batch: dict, device: torch.device, amp: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
        logits = model(images)
    probs = torch.sigmoid(logits.float())
    return probs, batch["mask"].to(device), batch["validity_mask"].to(device)


@torch.no_grad()
def select_thresholds_on_val(
    model: torch.nn.Module, val_loader: DataLoader, class_names: list[str], device: torch.device, amp: bool = True
) -> dict[str, dict[str, float]]:
    model.eval()
    sweep = ThresholdSweepAccumulator(class_names)
    for batch in val_loader:
        probs, target, validity = _predict(model, batch, device, amp)
        sweep.update(probs, target, validity)
    return sweep.best_thresholds()


class _PerImageMetrics:
    def __init__(self, class_names: list[str], thin_classes: list[str]) -> None:
        self.class_names, self.thin = class_names, set(thin_classes)
        self.boundary: dict[str, list[float]] = defaultdict(list)
        self.cldice: dict[str, list[float]] = defaultdict(list)
        self.inst_hits = {c: {b: 0 for b in SIZE_BUCKETS} for c in class_names}
        self.inst_totals = {c: {b: 0 for b in SIZE_BUCKETS} for c in class_names}

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """pred/gt: (C, H, W) bool, already validity-masked."""
        for c, name in enumerate(self.class_names):
            p, g = pred[c], gt[c]
            if not p.any() and not g.any():
                continue
            self.boundary[name].append(boundary_f1(p, g))
            if name in self.thin:
                self.cldice[name].append(hard_cl_dice(p, g))
            r = size_bucketed_detection_rate(p, g)
            for b in SIZE_BUCKETS:
                self.inst_totals[name][b] += r.num_gt_instances[b]
                self.inst_hits[name][b] += r.num_detected[b]


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    class_names: list[str],
    thin_classes: list[str],
    thresholds: dict[str, float],
    device: torch.device,
    amp: bool = True,
) -> dict[str, Any]:
    model.eval()
    thr = torch.tensor([thresholds[c] for c in class_names], device=device)
    overall = PixelMetricAccumulator(class_names, threshold=thr)
    per_source: dict[str, PixelMetricAccumulator] = {}
    per_image = _PerImageMetrics(class_names, thin_classes)

    for batch in loader:
        probs, target, validity = _predict(model, batch, device, amp)
        overall.update(probs, target, validity)
        for src in set(batch["source_dataset"]):
            sel = torch.tensor([s == src for s in batch["source_dataset"]], device=device)
            acc = per_source.setdefault(src, PixelMetricAccumulator(class_names, threshold=thr))
            acc.update(probs[sel], target[sel], validity[sel])

        pred_bin = ((probs >= thr.view(1, -1, 1, 1)).float() * validity).bool().cpu().numpy()
        gt_bin = (target * validity).bool().cpu().numpy()
        for i in range(pred_bin.shape[0]):
            per_image.update(pred_bin[i], gt_bin[i])

    return {"overall": overall.compute(), "per_source": {s: a.compute() for s, a in per_source.items()}, "per_image": per_image}


def results_to_dataframe(result: dict[str, Any], class_names: list[str], thresholds: dict[str, dict[str, float]], split: str) -> pd.DataFrame:
    pix, per_image = result["overall"], result["per_image"]
    rows = []
    for name in class_names:
        m = pix["per_class"][name]
        row = {
            "split": split, "scope": "overall", "class": name,
            "threshold": thresholds[name]["threshold"], "val_dice_at_threshold": thresholds[name]["val_dice"],
            "iou": m["iou"], "dice": m["dice"], "precision": m["precision"], "recall": m["recall"], "gt_pixels": m["gt_pixels"],
            "boundary_f1": float(np.mean(per_image.boundary[name])) if per_image.boundary[name] else float("nan"),
            "cldice": float(np.mean(per_image.cldice[name])) if per_image.cldice.get(name) else float("nan"),
            "n_images_boundary": len(per_image.boundary[name]),
        }
        for b in SIZE_BUCKETS:
            tot, hit = per_image.inst_totals[name][b], per_image.inst_hits[name][b]
            row[f"inst_recall_{b}"] = hit / tot if tot else float("nan")
            row[f"inst_n_gt_{b}"] = tot
        rows.append(row)
    aggs = ["macro", "frequency_weighted"]
    if any(pix["per_class"][n]["gt_pixels"] == 0 for n in class_names):  # dataset lacks some class: also give the fair average
        aggs.append("macro_present")
    for agg in aggs:
        rows.append({"split": split, "scope": "overall", "class": agg, **{k: pix[agg][k] for k in ("iou", "dice", "precision", "recall")}})
    for src, res in result["per_source"].items():
        for name in class_names:
            m = res["per_class"][name]
            rows.append({"split": split, "scope": f"source:{src}", "class": name, "threshold": thresholds[name]["threshold"],
                         "iou": m["iou"], "dice": m["dice"], "precision": m["precision"], "recall": m["recall"], "gt_pixels": m["gt_pixels"]})
        for agg in ("macro", "frequency_weighted"):
            rows.append({"split": split, "scope": f"source:{src}", "class": agg, **{k: res[agg][k] for k in ("iou", "dice", "precision", "recall")}})
    return pd.DataFrame(rows)


@torch.no_grad()
def sanity_check_gt_as_prediction(loader: DataLoader, class_names: list[str], device: torch.device, max_batches: int = 4) -> None:
    """Feeds ground truth back in as the 'prediction' through the same pixel + per-image code
    paths and asserts every class present in the sampled data scores ~1.0. If this fails, the bug
    is in the data/eval pipeline (mask conversion, validity handling, metric code), not the model —
    run it against real data before trusting any other number."""
    acc = PixelMetricAccumulator(class_names)
    per_image = _PerImageMetrics(class_names, thin_classes=[class_names[0]])
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        target = batch["mask"].to(device)
        validity = batch["validity_mask"].to(device)
        acc.update(target, target, validity)
        gt = (target * validity).bool().cpu().numpy()
        for j in range(gt.shape[0]):
            per_image.update(gt[j], gt[j])
    res = acc.compute()
    for name in class_names:
        m = res["per_class"][name]
        if m["gt_pixels"] > 0:
            assert abs(m["iou"] - 1.0) < 1e-4 and abs(m["dice"] - 1.0) < 1e-4, f"GT-as-prediction failed for {name}: {m}"
            assert np.mean(per_image.boundary[name]) > 0.999, f"boundary F1 GT-as-prediction failed for {name}"
            for b in SIZE_BUCKETS:
                if per_image.inst_totals[name][b]:
                    assert per_image.inst_hits[name][b] == per_image.inst_totals[name][b], f"instance detection GT-as-prediction failed for {name}/{b}"
