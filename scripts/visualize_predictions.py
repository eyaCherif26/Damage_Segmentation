"""Qualitative review: predictions of one or more models on images of your choice.

DATA — pick exactly one source (default: --split test):
  --split train|val|test                 the configured experiment data (with ground truth)
  --coco FILE --images DIR               ANY COCO file + image folder (with ground truth), e.g. new labelled data
        [--label_map YAML]                 maps its categories to the MVP classes; omitted -> its categories must
                                           already BE the MVP class names (e.g. the 5-class remapped files)
  --image_dir DIR [--recursive]          ANY folder of images WITHOUT labels (prediction-only panels)

RANDOMNESS: images are drawn at random and a fresh draw is made on EVERY run. The seed used is printed and saved
in selection.json; pass --seed N to reproduce a draw exactly.

Examples:
    # 12 new random test images
    python scripts/visualize_predictions.py \
        --model configs/experiments/effb0_task_pretrained_cif.yaml runs/effb0_task_pretrained_cif/best.pt \
        --num_images 12 --output_dir viz/effb0_cif

    # labelled data that is NOT in the configured splits
    python scripts/visualize_predictions.py --model <config> <checkpoint> \
        --coco path/to/annotations.json --images path/to/images --output_dir viz/new_labelled

    # unlabelled new images, two models side by side
    python scripts/visualize_predictions.py --model <cfgA> <ckptA> --model <cfgB> <ckptB> \
        --image_dir path/to/new/images --num_images 16 --output_dir viz/new_unlabelled

    # one model, several thresholds side by side (does a lower cutoff catch more of a rare class?)
    python scripts/visualize_predictions.py --model <config> <checkpoint> \
        --compare_thresholds 0.1 0.3 0.5 --output_dir viz/rust_threshold_check

THRESHOLDS — by default, per-model thresholds come from results/<experiment name>/thresholds.json (val-selected
by scripts/evaluate.py), falling back to 0.5 with a warning. Override, in increasing precedence:
  --model CONFIG CHECKPOINT VALUE_OR_FILE   a bare number (e.g. `0.3`) applies to every class of THAT model;
                                             a path loads a thresholds.json for that model, same as before.
  --fixed_threshold VALUE                   applies to every class of EVERY model (overrides all of the above).
  --class_threshold CLASS=VALUE ...         per-class override on top of whichever base above was used.
  --compare_thresholds V1 V2 ...            needs exactly one --model: renders that model's own base threshold
                                             plus one extra column per given value, all at once, so you can see
                                             the effect of the cutoff directly instead of guessing from numbers.

--only_classes CLASS [CLASS ...]          NOT retraining: renders only these classes. Normally every --model
                                           must share the exact same class list; with --only_classes, each model
                                           only needs to KNOW every requested class (its own full list may differ
                                           in content/order from the others') — the common subset is what's shown.
Only for sources WITH ground truth: --select worst|best (ranked by the first column's macro Dice), --source,
--require_class.
Images are shown at the evaluation resolution the model sees by default; for a large image this squeezes (whole_image)
or grid-tiles (native_crop) it exactly like the configured val/test split, which is NOT how scripts/infer_orthophoto.py
actually predicts on it. Pass --sliding_window (needs --coco or --image_dir, not --split) to instead render each
original image at full native resolution via inference/sliding_window.py::run_tiled_inference — the exact function
infer_orthophoto.py/predict_deployment.py use for real deployment (overlap-blended, one stitched probability map per
image) — regardless of resize_mode, same as evaluate.py's --sliding_window. One rendered panel per whole image, not
per tile: --num_images/--select worst|best/--max_scan then count/rank whole images, not tiles — and each candidate now
costs a full tiled-inference pass, so lower --max_scan a lot (e.g. 20-50, not the 600 default) when combined with
--select worst|best. --tile_overlap/--max_memory_gb/--allow_large_memory tune the sliding window itself.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset

import models
from configs.loader import load_experiment_config
from data.common_schema import rasterize_annotations
from data.datasets.combined_dataset import build_split_datasets
from data.datasets.external import build_labelled_dataset
from data.datasets.image_folder_dataset import ImageFolderDataset
from data.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform
from data.validity_mask import no_data_validity_mask, training_validity_mask
from inference.sliding_window import run_tiled_inference
from inference.visualize import (
    candidate_indices,
    denormalize,
    macro_dice_over_gt_classes,
    per_class_dice,
    pick_random,
    render_comparison,
    resolve_seed,
    stack_vertically,
)
from metrics.evaluation import load_thresholds, parse_threshold_overrides, resolve_class_subset


def load_model(config_path: Path, checkpoint: Path, device: torch.device):
    cfg = load_experiment_config(config_path)
    model = models.build_model(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    print(f"[viz] loaded {cfg.name} <- {checkpoint} (epoch {state.get('epoch', '?')})")
    return cfg, model.to(device).eval()


@torch.no_grad()
def predict(model, image: torch.Tensor, thresholds: torch.Tensor, device: torch.device, amp: bool,
           channel_indices: list[int] | None = None) -> np.ndarray:
    with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
        logits = model(image.unsqueeze(0).to(device))
    probs = torch.sigmoid(logits.float())[0]
    if channel_indices is not None:
        probs = probs[channel_indices]  # this model's own full output, reordered/filtered into `classes`' order
    return (probs >= thresholds.view(-1, 1, 1)).cpu().numpy()


@torch.no_grad()
def predict_tiled(
    model, image: np.ndarray, num_model_classes: int, tile_size: int, overlap: int, batch_size: int,
    device: torch.device, thresholds: torch.Tensor, channel_indices: list[int] | None,
    max_memory_gb: float = 8.0, allow_large_memory: bool = False,
) -> np.ndarray:
    """Like predict(), but for a full native-resolution image (H, W, 3) uint8 RGB instead of one
    pre-transformed dataset item: runs inference/sliding_window.py::run_tiled_inference — the exact
    function scripts/infer_orthophoto.py/predict_deployment.py use for real deployment — and
    thresholds the stitched result, instead of one forward pass at the training tile size."""
    prob_map = run_tiled_inference(model, image, num_model_classes, tile_size, overlap, batch_size, device,
                                   max_memory_gb=max_memory_gb, allow_large_memory=allow_large_memory)
    if channel_indices is not None:
        prob_map = prob_map[channel_indices]
    return prob_map >= thresholds.view(-1, 1, 1).cpu().numpy()


def sample_raw(raw_ds, idx: int, has_gt: bool):
    """Like indexing a dataset item, but un-resized/un-tiled: the raw native-resolution image (H, W,
    3) uint8 RGB (what predict_tiled/run_tiled_inference expects), full-resolution ground truth
    (rasterized directly, not through a dataset __getitem__ that would resize/crop it), and full
    validity — used only when --sliding_window is set. `raw_ds` is either a CocoMultiLabelSegDataset
    built with force_whole_image=True (has_gt=True) or an ImageFolderDataset (has_gt=False)."""
    if has_gt:
        image_id = raw_ds.image_ids[idx]
        info = raw_ds.images_by_id[image_id]
        image = raw_ds._load_image(info["file_name"])
        height, width = info["height"], info["width"]
        anns = raw_ds.annotations_by_image.get(image_id, [])
        full_mask = rasterize_annotations(anns, height, width, raw_ds.category_id_to_channel, len(raw_ds.mvp_classes))
        gt = full_mask > 0.5
        validity = (no_data_validity_mask(image, max_pixels=2_000_000) if raw_ds.detect_no_data
                   else training_validity_mask(height, width))[0] > 0.5
        return image, gt, validity, raw_ds.source_name, image_id
    path = raw_ds.files[idx]
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    validity = (no_data_validity_mask(image, max_pixels=2_000_000)[0] if raw_ds.detect_no_data
               else np.ones((height, width), np.float32)) > 0.5
    image_id = str(path.relative_to(raw_ds.root).with_suffix("")).replace("/", "__")
    return image, None, validity, raw_ds.source_name, image_id


def build_dataset(args, base_cfg, classes, eval_tf):
    """Returns (dataset, has_ground_truth, description). With args.sliding_window, `dataset` is the
    RAW (un-resized, un-tiled) dataset instead — see sample_raw(), used for rendering instead of
    plain indexing whenever args.sliding_window is set."""
    native = base_cfg.data.resize_mode == "native_crop"
    if args.image_dir:
        if native and not args.sliding_window:
            raise SystemExit(
                "This model was trained in native_crop mode (fixed pixel scale, tiles). --image_dir would resize whole "
                "images and show predictions at the wrong scale. Pass --sliding_window (full native-resolution tiled "
                "inference, matching scripts/infer_orthophoto.py) or use that script directly."
            )
        if args.sliding_window:
            ds = ImageFolderDataset(args.image_dir, len(classes), transform=None, recursive=args.recursive,
                                    detect_no_data=base_cfg.data.no_data_validity)
            return ds, False, f"unlabelled images in {args.image_dir} (sliding-window inference, full native resolution)"
        ds = ImageFolderDataset(args.image_dir, len(classes), transform=eval_tf, recursive=args.recursive,
                                detect_no_data=base_cfg.data.no_data_validity)
        return ds, False, f"unlabelled images in {args.image_dir}"
    if args.coco:
        if args.sliding_window:
            ds, info = build_labelled_dataset(args.coco, args.images, base_cfg.data, None, args.label_map, force_whole_image=True)
            return ConcatDataset([ds]), True, f"COCO {args.coco} + images {args.images} (sliding-window inference, full native resolution)"
        ds, info = build_labelled_dataset(args.coco, args.images, base_cfg.data, eval_tf, args.label_map)
        tiles = f" (tiles at pixel_scale {base_cfg.data.pixel_scale})" if native else ""
        return ConcatDataset([ds]), True, f"COCO {args.coco} + images {args.images}{tiles}"
    if args.sliding_window:
        raise SystemExit("--sliding_window needs --coco or --image_dir, not --split — the configured split's own protocol is unaffected by it")
    split = args.split or "test"
    return build_split_datasets(base_cfg.data, eval_tf, eval_tf)[split], True, f"configured {split} split"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", action="append", required=True, metavar=("CONFIG", "CHECKPOINT"),
                   help="CONFIG CHECKPOINT [VALUE_OR_THRESHOLDS_JSON]; repeat for several models")
    src = p.add_argument_group("data source (choose one)")
    src.add_argument("--split", choices=["train", "val", "test"], default=None, help="configured split (default: test)")
    src.add_argument("--coco", default=None, help="COCO annotation file of any dataset (needs --images)")
    src.add_argument("--images", default=None, help="image folder belonging to --coco")
    src.add_argument("--label_map", default=None, help="YAML mapping the --coco categories to MVP classes (or null)")
    src.add_argument("--image_dir", default=None, help="folder of images WITHOUT labels")
    src.add_argument("--recursive", action="store_true", help="with --image_dir: also search sub-folders")
    src.add_argument("--sliding_window", action="store_true",
                     help="Render each original image at full native resolution via sliding-window, overlap-blended "
                          "inference (inference/sliding_window.py::run_tiled_inference — the exact function "
                          "scripts/infer_orthophoto.py/predict_deployment.py use for real deployment) instead of at "
                          "the model's own evaluation resolution (a whole-image resize, or a grid tile for a "
                          "native_crop model) — regardless of resize_mode, same as scripts/evaluate.py's "
                          "--sliding_window. Needs --coco or --image_dir, not --split. One panel per whole ORIGINAL "
                          "image, not per tile: --num_images/--select/--max_scan then count/rank whole images.")
    src.add_argument("--tile_overlap", type=int, default=128, help="--sliding_window only: tile overlap in pixels")
    src.add_argument("--tile_batch_size", type=int, default=4, help="--sliding_window only: tiles per forward pass (matches infer_orthophoto.py's default)")
    src.add_argument("--max_memory_gb", type=float, default=8.0, help="--sliding_window only: VRAM budget guard")
    src.add_argument("--allow_large_memory", action="store_true")
    thr = p.add_argument_group("threshold overrides (see THRESHOLDS in the module docstring)")
    thr.add_argument("--fixed_threshold", type=float, default=None, help="apply this one value to every class of EVERY model")
    thr.add_argument("--class_threshold", nargs="+", default=[], metavar="CLASS=VALUE",
                     help="override specific classes on top of the base method, e.g. --class_threshold Rust=0.1")
    thr.add_argument("--compare_thresholds", nargs="+", type=float, default=[], metavar="VALUE",
                     help="render one extra column per value, all at this scalar threshold for every class; needs exactly one --model")
    p.add_argument("--only_classes", nargs="+", default=None, metavar="CLASS",
                   help="render only these classes; relaxes the 'all models share the same class list' rule to "
                        "'every model knows at least these classes' (see THRESHOLDS in the module docstring)")
    p.add_argument("--num_images", type=int, default=12)
    p.add_argument("--seed", type=int, default=None, help="omit for a fresh random draw each run; set to reproduce one")
    p.add_argument("--select", choices=["random", "worst", "best"], default="random",
                   help="worst/best rank by the first model's (or, with --compare_thresholds, the base column's) macro Dice (needs ground truth)")
    p.add_argument("--source", default=None, help="only images from this source, e.g. cif_coco or dacl10k (ground-truth sources)")
    p.add_argument("--require_class", nargs="+", default=[], help="only images whose ground truth contains all these classes")
    p.add_argument("--max_scan", type=int, default=600, help="for worst/best: score at most this many random candidates")
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--output_dir", required=True, type=Path)
    args = p.parse_args()

    chosen_sources = [bool(args.split), bool(args.coco), bool(args.image_dir)]
    if sum(chosen_sources) > 1:
        p.error("choose only one data source: --split, --coco/--images, or --image_dir")
    if args.coco and not args.images:
        p.error("--coco needs --images (the folder holding that COCO file's images)")
    if (args.images or args.label_map) and not args.coco:
        p.error("--images/--label_map only apply together with --coco")
    for spec in args.model:
        if not 2 <= len(spec) <= 3:
            p.error("--model takes CONFIG CHECKPOINT [VALUE_OR_THRESHOLDS_JSON]")
    if args.compare_thresholds and len(args.model) != 1:
        p.error("--compare_thresholds needs exactly one --model (it compares thresholds of ONE model, not several models)")
    if args.fixed_threshold is not None and not (0.0 <= args.fixed_threshold <= 1.0):
        p.error(f"--fixed_threshold must be in [0, 1], got {args.fixed_threshold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = []
    for spec in args.model:
        cfg, model = load_model(Path(spec[0]), Path(spec[1]), device)
        loaded.append((cfg, model, spec[2] if len(spec) == 3 else None))

    base_cfg = loaded[0][0]
    all_classes = list(base_cfg.data.mvp_classes)  # this model's own full class list; used for GT/dataset construction
    for cfg, _, _ in loaded[1:]:
        if cfg.data.image_size != base_cfg.data.image_size:
            raise SystemExit("All models must share image_size to be compared on identical images.")
        if not args.only_classes and list(cfg.data.mvp_classes) != all_classes:
            raise SystemExit("All models must share the same class list and image_size to be compared on identical "
                              "images (or pass --only_classes to compare on a common subset of classes instead).")
    for c in args.require_class:
        if c not in all_classes:
            raise SystemExit(f"--require_class {c!r} is not one of {all_classes}")

    # classes: what gets rendered/scored, in the base model's own class order (stable regardless of how
    # --only_classes was typed). base_channel_indices: where those classes sit in the base model's own
    # (== all_classes') output — always None here since `classes` IS all_classes' own order already,
    # kept for symmetry with model_channel_indices below (every model, base included, gets a real slice).
    classes, base_channel_indices = resolve_class_subset(all_classes, args.only_classes)
    if args.only_classes:
        print(f"[viz] --only_classes: rendering only {classes} (of the base model's {all_classes})")
    class_overrides = parse_threshold_overrides(args.class_threshold, classes) if args.class_threshold else {}

    # Each model's own channel indices for `classes`, in `classes`' order (not that model's own order —
    # models may list their classes differently; render_comparison needs every column aligned the same way).
    model_channel_indices = []
    for cfg, _, _ in loaded:
        model_all = list(cfg.data.mvp_classes)
        missing = [c for c in classes if c not in model_all]
        if missing:
            raise SystemExit(f"--only_classes {missing} not in {cfg.name}'s classes {model_all}")
        model_channel_indices.append([model_all.index(c) for c in classes])
    model_num_classes = [len(cfg.data.mvp_classes) for cfg, _, _ in loaded]  # --sliding_window: this model's own full output width

    def base_threshold_tensor(cfg, explicit: str | None) -> torch.Tensor:
        """Per-model base threshold, before --fixed_threshold/--class_threshold overrides: a bare
        number in `explicit` applies to every class of this model; a path loads a thresholds.json;
        neither given falls back to results/<experiment>/thresholds.json, then to 0.5 with a warning."""
        if explicit is not None:
            try:
                value = float(explicit)
            except ValueError:
                pass
            else:
                print(f"[viz] {cfg.name}: fixed threshold {value} for every class")
                return torch.full((len(classes),), value, device=device)
        path = Path(explicit) if explicit is not None else Path("results") / cfg.name / "thresholds.json"
        th = load_thresholds(path, classes)
        if th is None:
            print(f"[viz] WARNING: no thresholds at {path}; using 0.5 for {cfg.name}. Run scripts/evaluate.py first for tuned ones.")
            th = {c: 0.5 for c in classes}
        else:
            print(f"[viz] {cfg.name}: thresholds from {path}: {th}")
        return torch.tensor([th[c] for c in classes], device=device)

    def apply_overrides(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone()
        if args.fixed_threshold is not None:
            tensor.fill_(args.fixed_threshold)
        for c, v in class_overrides.items():
            tensor[classes.index(c)] = v
        return tensor

    threshold_tensors = [apply_overrides(base_threshold_tensor(cfg, explicit)) for cfg, _, explicit in loaded]
    if args.fixed_threshold is not None:
        print(f"[viz] --fixed_threshold {args.fixed_threshold} applied to every class of every model, overriding the above")
    if class_overrides:
        print(f"[viz] --class_threshold overrides applied on top: {class_overrides}")

    # "Columns" to render: normally one per --model; with --compare_thresholds, one model's own base
    # column plus one extra column per given value (all classes, same scalar) — render_comparison()
    # already takes an arbitrary {label: prediction} dict, so this reuses it unchanged. 5th/6th
    # elements: this column's model's own channel indices for `classes` (model_channel_indices
    # above) and its own full output width (model_num_classes above, --sliding_window only).
    if args.compare_thresholds:
        cfg0, model0, explicit0 = loaded[0]
        idx0, nc0 = model_channel_indices[0], model_num_classes[0]
        columns = [(f"{cfg0.name} (base)", model0, threshold_tensors[0], cfg0.training.amp, idx0, nc0)]
        for v in args.compare_thresholds:
            if not (0.0 <= v <= 1.0):
                p.error(f"--compare_thresholds values must be in [0, 1], got {v}")
            columns.append((f"th={v:g}", model0, torch.full((len(classes),), v, device=device), cfg0.training.amp, idx0, nc0))
    else:
        columns = [(cfg.name, model, th, cfg.training.amp, idx, nc)
                  for (cfg, model, _), th, idx, nc in zip(loaded, threshold_tensors, model_channel_indices, model_num_classes)]

    ds, has_gt, description = build_dataset(args, base_cfg, all_classes, build_eval_transform(base_cfg.data.image_size))
    if not has_gt:
        for flag, bad in (("--select worst/best", args.select != "random"),
                          ("--require_class", bool(args.require_class)), ("--source", bool(args.source))):
            if bad:
                raise SystemExit(f"{flag} needs ground truth, but --image_dir images have none.")

    candidates = candidate_indices(ds, all_classes, source=args.source, require_classes=args.require_class) if has_gt else list(range(len(ds)))
    if not candidates:
        raise SystemExit("No images match --source/--require_class.")
    seed = resolve_seed(args.seed)
    print(f"[viz] data: {description}: {len(candidates)} candidate images | random seed {seed} "
          f"({'given' if args.seed is not None else 'fresh — pass --seed ' + str(seed) + ' to reproduce this draw'})")

    # --sliding_window: `ds` is the raw, un-resized/un-tiled dataset (a ConcatDataset of one
    # CocoMultiLabelSegDataset for --coco, or a bare ImageFolderDataset for --image_dir); unwrap the
    # former the same way it was wrapped in build_dataset, purely so sample_raw sees the real object.
    raw_ds = ds.datasets[0] if (args.sliding_window and has_gt) else ds

    def sample_for(index: int):
        """Returns (model_input, display_image, gt, valid, source_dataset, image_id).
        Non-tiled: model_input is the normalized CHW tensor predict() expects, display_image is it
        denormalized back to uint8 HWC for rendering. --sliding_window: both are the same raw uint8
        HWC image — predict_tiled()/run_tiled_inference do their own normalization internally, and
        render_comparison already expects uint8 HWC, so no denormalize step is needed at all."""
        if args.sliding_window:
            image, gt, valid, source_dataset, image_id = sample_raw(raw_ds, index, has_gt)
            if gt is not None and base_channel_indices is not None:
                gt = gt[base_channel_indices]
            return image, image, gt, valid, source_dataset, image_id
        item = ds[index]
        image = denormalize(item["image"].numpy(), IMAGENET_MEAN, IMAGENET_STD)
        gt = None
        if has_gt:
            gt = item["mask"].numpy() > 0.5
            if base_channel_indices is not None:
                gt = gt[base_channel_indices]
        return item["image"], image, gt, item["validity_mask"].numpy()[0] > 0.5, item["source_dataset"], item["image_id"]

    def do_predict(model, model_input, th, amp, col_idx, num_model_classes) -> np.ndarray:
        if args.sliding_window:
            return predict_tiled(model, model_input, num_model_classes, base_cfg.data.image_size, args.tile_overlap,
                                 args.tile_batch_size, device, th, col_idx, args.max_memory_gb, args.allow_large_memory)
        return predict(model, model_input, th, device, amp, col_idx)

    if args.select == "random":
        chosen = pick_random(candidates, args.num_images, seed)
    else:
        pool = pick_random(candidates, args.max_scan, seed)
        base_label, base_model, base_th, base_amp, base_idx, base_nc = columns[0]
        print(f"[viz] ranking {len(pool)} images by {base_label} macro Dice to pick the {args.select} {args.num_images} ..."
              + (" (each candidate costs a full sliding-window pass — consider a lower --max_scan)" if args.sliding_window else ""))
        scored = []
        for idx in pool:
            model_input, _, gt, valid, _, _ = sample_for(idx)
            pred = do_predict(base_model, model_input, base_th, base_amp, base_idx, base_nc)
            score = macro_dice_over_gt_classes(per_class_dice(pred, gt, valid), gt, valid)
            if not np.isnan(score):
                scored.append((score, idx))
        scored.sort(reverse=(args.select == "best"))
        chosen = [i for _, i in scored[: args.num_images]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stale = [f for f in args.output_dir.glob("[0-9][0-9]_*.png")]
    if stale:
        print(f"[viz] WARNING: {args.output_dir} already holds {len(stale)} figure(s) from an earlier run; new ones are written "
              "alongside and may overwrite some. Use a fresh --output_dir (or delete the old files) to keep runs separate.")
    rows, written, selection = [], [], []
    for rank, idx in enumerate(chosen):
        model_input, image, gt, valid, source_dataset, image_id = sample_for(idx)
        preds, scores = {}, {}
        for label, model, th, amp, col_idx, num_model_classes in columns:
            preds[label] = do_predict(model, model_input, th, amp, col_idx, num_model_classes)
            scores[label] = macro_dice_over_gt_classes(per_class_dice(preds[label], gt, valid), gt, valid) if has_gt else float("nan")
        title = f"{source_dataset}  {image_id}"
        fig = render_comparison(image, gt, preds, valid, classes, title, dpi=args.dpi)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(image_id))
        path = args.output_dir / f"{rank:02d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', source_dataset)}_{safe_id}.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
        rows.append((path.name, scores))
        selection.append({"rank": rank, "index": int(idx), "source_dataset": source_dataset, "image_id": str(image_id), "file": path.name})

    cv2.imwrite(str(args.output_dir / "overview.png"), stack_vertically([cv2.imread(str(w)) for w in written]))
    (args.output_dir / "selection.json").write_text(json.dumps({
        "data": description, "seed": seed, "seed_was_given": args.seed is not None, "select": args.select,
        "models": [{"config": s[0], "checkpoint": s[1]} for s in args.model], "images": selection}, indent=2))

    names = [label for label, _, _, _, _, _ in columns]
    if has_gt:
        print(f"\n{'image':46s} " + " ".join(f"{n[:22]:>22s}" for n in names))
        for fname, sc in rows:
            print(f"{fname:46s} " + " ".join(f"{sc[n]:22.3f}" for n in names))
    print(f"\n[viz] wrote {len(written)} figures + overview.png + selection.json to {args.output_dir} (seed {seed})")


if __name__ == "__main__":
    main()
