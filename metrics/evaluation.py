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

from data.common_schema import rasterize_annotations
from data.datasets.multilabel_seg_dataset import CocoMultiLabelSegDataset
from data.validity_mask import no_data_validity_mask, training_validity_mask
from inference.sliding_window import run_tiled_inference
from metrics.boundary_metrics import boundary_f1, hard_cl_dice
from metrics.instance_metrics import SIZE_BUCKETS, size_bucketed_detection_rate
from metrics.pixel_metrics import PixelMetricAccumulator, ThresholdSweepAccumulator


def parse_json_with_helpful_error(path: Path) -> Any:
    """json.loads with a specific hint for the one mistake that keeps recurring in this project:
    hand-editing a thresholds.json by commenting out an old block with '//' and pasting new values
    below it — plain JSON has no comment syntax, so json.loads fails outright ('Expecting value:
    line 1 column 1') with no clue why. Shared by load_thresholds below, scripts/evaluate.py,
    scripts/export_deployment_model.py, and scripts/predict_deployment.py."""
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        hint = " JSON has no comment syntax — remove any '//' lines (and the block they precede if unused) and re-save." if "//" in text else ""
        raise ValueError(f"{path} is not valid JSON: {e}.{hint}") from e


def load_thresholds(path: str | Path, class_names: list[str]) -> dict[str, float] | None:
    """Reads the `thresholds.json` written by scripts/evaluate.py (`{class: {"threshold": t, "val_dice": d}}`).
    Returns None if the file does not exist; raises if it exists but does not cover every class, since a
    silently partial threshold file would give misleading predictions."""
    path = Path(path)
    if not path.exists():
        return None
    raw = parse_json_with_helpful_error(path)
    missing = [c for c in class_names if c not in raw]
    if missing:
        raise KeyError(f"{path} has no threshold for class(es) {missing}; it covers {sorted(raw)}. "
                       "Re-run scripts/evaluate.py with this experiment's class list.")
    return {c: float(raw[c]["threshold"] if isinstance(raw[c], dict) else raw[c]) for c in class_names}


def parse_threshold_overrides(pairs: list[str], class_names: list[str]) -> dict[str, float]:
    """Parses `--class_threshold` CLI values like `["Cracks=0.3", "Rust=0.1"]` into
    `{"Cracks": 0.3, "Rust": 0.1}`, shared by scripts/evaluate.py and scripts/visualize_predictions.py.
    Validates eagerly (unknown class name, non-numeric value, out-of-[0,1] value) so a typo fails
    loudly at argument-parsing time rather than silently mis-thresholding a class."""
    out: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--class_threshold entries must be CLASS=VALUE, got {pair!r}")
        name, _, value = pair.partition("=")
        if name not in class_names:
            raise ValueError(f"--class_threshold: {name!r} is not one of {class_names}")
        try:
            v = float(value)
        except ValueError:
            raise ValueError(f"--class_threshold: {value!r} is not a number (for class {name})")
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"--class_threshold: {name}={v} must be in [0, 1]")
        out[name] = v
    return out


def resolve_class_subset(full_classes: list[str], requested: list[str] | None) -> tuple[list[str], list[int] | None]:
    """`--only_classes`-style filter shared by scripts/evaluate.py, scripts/visualize_predictions.py
    and scripts/infer_orthophoto.py: NOT retraining — a post-hoc filter on an already-trained
    checkpoint's known classes (`full_classes`, its `cfg.data.mvp_classes`).

    `requested=None` -> `(list(full_classes), None)`: no filtering, no slicing overhead.
    Otherwise validates `requested` is a subset of `full_classes` (raises naming any name that
    isn't, before any dataset/model work) and returns the subset **in `full_classes`' own order**
    (not the order `requested` was typed in) plus the channel indices to slice the model's raw
    per-checkpoint output by — this keeps colors/CSV columns/thresholds.json keys stable whether
    or not a filter is used, regardless of how `--only_classes` was typed."""
    if requested is None:
        return list(full_classes), None
    unknown = [c for c in requested if c not in full_classes]
    if unknown:
        raise ValueError(f"not in this checkpoint's classes {full_classes}: {unknown}")
    wanted = set(requested)
    indices = [i for i, c in enumerate(full_classes) if c in wanted]
    return [full_classes[i] for i in indices], indices


@torch.no_grad()
def _predict(
    model: torch.nn.Module, batch: dict, device: torch.device, amp: bool, channel_indices: list[int] | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
        logits = model(images)
    probs = torch.sigmoid(logits.float())
    target = batch["mask"].to(device)
    if channel_indices is not None:
        idx = torch.tensor(channel_indices, device=probs.device)
        probs, target = probs.index_select(1, idx), target.index_select(1, idx)
    return probs, target, batch["validity_mask"].to(device)


@torch.no_grad()
def select_thresholds_on_val(
    model: torch.nn.Module, val_loader: DataLoader, class_names: list[str], device: torch.device, amp: bool = True,
    channel_indices: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    model.eval()
    sweep = ThresholdSweepAccumulator(class_names)
    for batch in val_loader:
        probs, target, validity = _predict(model, batch, device, amp, channel_indices)
        sweep.update(probs, target, validity)
    return sweep.best_thresholds()


class _PerImageMetrics:
    def __init__(self, class_names: list[str], thin_classes: list[str], instance_iou_threshold: float = 0.5) -> None:
        """`instance_iou_threshold`: how much a predicted connected component must overlap a GT
        instance (IoU) to count that instance as "detected" for inst_recall_*. The default (0.5)
        is a strict shape/size match — for a thin, curved structure like a crack, a prediction that
        a human would call "found it" (roughly the right place, plausible width) routinely falls
        well under 0.5 IoU, which is why inst_recall_* can read exactly 0 even when the pixel-level
        Dice/recall above it are modestly positive and the overlay looks reasonable by eye. Lower
        this (e.g. 0.1, or scripts/evaluate.py's --detection_iou_threshold) for a client-facing
        "did it flag the defect at all" number — report it ALONGSIDE the strict pixel/boundary
        metrics, not instead of them; it answers a different question (presence vs. precise extent)."""
        self.class_names, self.thin = class_names, set(thin_classes)
        self.instance_iou_threshold = instance_iou_threshold
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
            r = size_bucketed_detection_rate(p, g, iou_threshold=self.instance_iou_threshold)
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
    channel_indices: list[int] | None = None,
    instance_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    thr = torch.tensor([thresholds[c] for c in class_names], device=device)
    overall = PixelMetricAccumulator(class_names, threshold=thr)
    per_source: dict[str, PixelMetricAccumulator] = {}
    per_image = _PerImageMetrics(class_names, thin_classes, instance_iou_threshold)

    for batch in loader:
        probs, target, validity = _predict(model, batch, device, amp, channel_indices)
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


@torch.no_grad()
def _iter_tiled_predictions(
    model: torch.nn.Module,
    dataset: CocoMultiLabelSegDataset,
    tile_size: int,
    overlap: int,
    batch_size: int,
    device: torch.device,
    max_memory_gb: float,
    allow_large_memory: bool,
    channel_indices: list[int] | None,
    image_ids: list | None = None,
):
    """Yields (probs, target, validity) — each a (1, C, H, W)/(1, 1, H, W) tensor at that image's
    own full native resolution — one whole image at a time via sliding-window, overlap-blended
    inference (inference/sliding_window.py::run_tiled_inference — the exact function
    scripts/infer_orthophoto.py / scripts/predict_deployment.py use for real deployment). Shared by
    evaluate_external_with_tiled_inference and select_thresholds_with_tiled_inference so both use
    the exact same real-inference path, not two subtly different reimplementations of it.

    `dataset` must be a CocoMultiLabelSegDataset (see data/datasets/external.py::build_labelled_dataset's
    `force_whole_image=True`) — its bookkeeping (`images_by_id`/`annotations_by_image`/
    `category_id_to_channel`/`_load_image`) is used directly, un-resized; its own `__getitem__` is
    never called here (that would resize/normalize the image, exactly what this path avoids).
    `image_ids`, if given, restricts iteration to only those images (e.g. one side of a
    calibration/report split) — default (None) is every image in the dataset.
    """
    model.eval()
    num_model_classes = len(dataset.mvp_classes)  # the model's own full output width
    for image_id in (image_ids if image_ids is not None else dataset.image_ids):
        info = dataset.images_by_id[image_id]
        image = dataset._load_image(info["file_name"])  # raw uint8 RGB, full native resolution
        height, width = info["height"], info["width"]
        anns = dataset.annotations_by_image.get(image_id, [])
        full_mask = rasterize_annotations(anns, height, width, dataset.category_id_to_channel, num_model_classes)
        validity = (no_data_validity_mask(image, max_pixels=2_000_000) if dataset.detect_no_data
                   else training_validity_mask(height, width))

        prob_map = run_tiled_inference(model, image, num_model_classes, tile_size, overlap, batch_size, device,
                                       max_memory_gb=max_memory_gb, allow_large_memory=allow_large_memory)
        if channel_indices is not None:
            prob_map, full_mask = prob_map[channel_indices], full_mask[channel_indices]

        yield (torch.from_numpy(prob_map).unsqueeze(0).float(),
              torch.from_numpy(full_mask).unsqueeze(0).float(),
              torch.from_numpy(validity).unsqueeze(0).float())


def evaluate_external_with_tiled_inference(
    model: torch.nn.Module,
    dataset: CocoMultiLabelSegDataset,
    class_names: list[str],
    thin_classes: list[str],
    thresholds: dict[str, float],
    device: torch.device,
    tile_size: int,
    overlap: int = 128,
    batch_size: int = 4,
    max_memory_gb: float = 8.0,
    allow_large_memory: bool = False,
    channel_indices: list[int] | None = None,
    image_ids: list | None = None,
    instance_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Like evaluate_split, but scores each image via a single stitched full-resolution
    probability map from sliding-window inference (see _iter_tiled_predictions), instead of the
    non-overlapping fixed-tile grid `data.resize_mode: native_crop` evaluation normally uses
    (data/datasets/crop_dataset.py, mode="grid"). That grid hard-cuts a defect at tile boundaries
    with no blending between tiles; this reproduces what the model actually sees in production on
    an image larger than its training tile size. Iterates one whole image at a time (each may be a
    different size, so — unlike evaluate_split — there is no outer DataLoader batching;
    `batch_size` instead controls how many TILES of one image go through the model per forward
    pass, same meaning as in run_tiled_inference/infer_orthophoto.py). `image_ids`, if given,
    scores only those images (e.g. a held-out reporting split — see select_thresholds_with_tiled_inference).
    """
    thr = torch.tensor([thresholds[c] for c in class_names])
    overall = PixelMetricAccumulator(class_names, threshold=thr)
    per_image = _PerImageMetrics(class_names, thin_classes, instance_iou_threshold)

    for probs_t, target_t, validity_t in _iter_tiled_predictions(
        model, dataset, tile_size, overlap, batch_size, device, max_memory_gb, allow_large_memory, channel_indices, image_ids
    ):
        overall.update(probs_t, target_t, validity_t)
        pred_bin = ((probs_t >= thr.view(1, -1, 1, 1)).float() * validity_t).bool().numpy()[0]
        gt_bin = (target_t * validity_t).bool().numpy()[0]
        per_image.update(pred_bin, gt_bin)

    return {"overall": overall.compute(), "per_source": {dataset.source_name: overall.compute()}, "per_image": per_image}


def select_thresholds_with_tiled_inference(
    model: torch.nn.Module,
    dataset: CocoMultiLabelSegDataset,
    class_names: list[str],
    device: torch.device,
    tile_size: int,
    overlap: int = 128,
    batch_size: int = 4,
    max_memory_gb: float = 8.0,
    allow_large_memory: bool = False,
    channel_indices: list[int] | None = None,
    image_ids: list | None = None,
) -> dict[str, dict[str, float]]:
    """Like select_thresholds_on_val, but sweeps thresholds using sliding-window tiled inference's
    full-resolution probability maps (see _iter_tiled_predictions) instead of fixed-size batches —
    for scripts/evaluate.py's --retune_on_new_data/--calibration_fraction when the checkpoint is
    being evaluated with --sliding_window. `image_ids`, if given (the calibration split), restricts
    the sweep to only those images; the reporting split is then scored separately and never seen here."""
    sweep = ThresholdSweepAccumulator(class_names)
    for probs_t, target_t, validity_t in _iter_tiled_predictions(
        model, dataset, tile_size, overlap, batch_size, device, max_memory_gb, allow_large_memory, channel_indices, image_ids
    ):
        sweep.update(probs_t, target_t, validity_t)
    return sweep.best_thresholds()


def split_images_for_calibration(image_ids: list, calibration_fraction: float, seed: int) -> tuple[set, set]:
    """Splits a list of (possibly non-unique — duplicates are ignored) image ids into
    (calibration_ids, report_ids): the calibration subset is used ONLY to select thresholds, the
    report subset is the genuinely held-out set metrics get computed on — same discipline as the
    configured val/test split, just applied to a new dataset. Deterministic given the same seed."""
    if not (0.0 < calibration_fraction < 1.0):
        raise ValueError(f"calibration_fraction must be in (0, 1), got {calibration_fraction}")
    unique = sorted(set(image_ids), key=str)  # de-duplicate + deterministic order before shuffling
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique), generator=g).tolist()
    n_calib = max(1, round(len(unique) * calibration_fraction))
    if n_calib >= len(unique):
        raise ValueError(
            f"calibration_fraction={calibration_fraction} leaves no images for the held-out reporting set "
            f"({len(unique)} unique images total) — lower it, or supply more images."
        )
    calib = {unique[i] for i in perm[:n_calib]}
    report = set(unique) - calib
    return calib, report


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
