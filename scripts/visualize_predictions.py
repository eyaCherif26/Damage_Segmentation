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

`--model CONFIG CHECKPOINT [THRESHOLDS_JSON]`; thresholds default to results/<experiment name>/thresholds.json
(from scripts/evaluate.py, chosen on val) and fall back to 0.5 with a warning.
Only for sources WITH ground truth: --select worst|best (ranked by the first model's macro Dice), --source,
--require_class.
Images are shown at the evaluation resolution the model sees; for large orthophotos use scripts/infer_orthophoto.py.
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
from data.datasets.combined_dataset import build_split_datasets
from data.datasets.external import build_labelled_dataset
from data.datasets.image_folder_dataset import ImageFolderDataset
from data.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform
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
from metrics.evaluation import load_thresholds


def load_model(config_path: Path, checkpoint: Path, device: torch.device):
    cfg = load_experiment_config(config_path)
    model = models.build_model(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    print(f"[viz] loaded {cfg.name} <- {checkpoint} (epoch {state.get('epoch', '?')})")
    return cfg, model.to(device).eval()


@torch.no_grad()
def predict(model, image: torch.Tensor, thresholds: torch.Tensor, device: torch.device, amp: bool) -> np.ndarray:
    with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
        logits = model(image.unsqueeze(0).to(device))
    probs = torch.sigmoid(logits.float())[0]
    return (probs >= thresholds.view(-1, 1, 1)).cpu().numpy()


def build_dataset(args, base_cfg, classes, eval_tf):
    """Returns (dataset, has_ground_truth, description)."""
    native = base_cfg.data.resize_mode == "native_crop"
    if args.image_dir:
        if native:
            raise SystemExit(
                "This model was trained in native_crop mode (fixed pixel scale, tiles). --image_dir would resize whole "
                "images and show predictions at the wrong scale. Use scripts/infer_orthophoto.py (tiled inference) with "
                f"--scale chosen for your images (training used pixel_scale {base_cfg.data.pixel_scale} of its photos' native pixels)."
            )
        ds = ImageFolderDataset(args.image_dir, len(classes), transform=eval_tf, recursive=args.recursive,
                                detect_no_data=base_cfg.data.no_data_validity)
        return ds, False, f"unlabelled images in {args.image_dir}"
    if args.coco:
        ds, info = build_labelled_dataset(args.coco, args.images, base_cfg.data, eval_tf, args.label_map)
        tiles = f" (tiles at pixel_scale {base_cfg.data.pixel_scale})" if native else ""
        return ConcatDataset([ds]), True, f"COCO {args.coco} + images {args.images}{tiles}"
    split = args.split or "test"
    return build_split_datasets(base_cfg.data, eval_tf, eval_tf)[split], True, f"configured {split} split"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", action="append", required=True, metavar=("CONFIG", "CHECKPOINT"),
                   help="CONFIG CHECKPOINT [THRESHOLDS_JSON]; repeat for several models")
    src = p.add_argument_group("data source (choose one)")
    src.add_argument("--split", choices=["train", "val", "test"], default=None, help="configured split (default: test)")
    src.add_argument("--coco", default=None, help="COCO annotation file of any dataset (needs --images)")
    src.add_argument("--images", default=None, help="image folder belonging to --coco")
    src.add_argument("--label_map", default=None, help="YAML mapping the --coco categories to MVP classes (or null)")
    src.add_argument("--image_dir", default=None, help="folder of images WITHOUT labels")
    src.add_argument("--recursive", action="store_true", help="with --image_dir: also search sub-folders")
    p.add_argument("--num_images", type=int, default=12)
    p.add_argument("--seed", type=int, default=None, help="omit for a fresh random draw each run; set to reproduce one")
    p.add_argument("--select", choices=["random", "worst", "best"], default="random",
                   help="worst/best rank by the first model's macro Dice (needs ground truth)")
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
            p.error("--model takes CONFIG CHECKPOINT [THRESHOLDS_JSON]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = []
    for spec in args.model:
        cfg, model = load_model(Path(spec[0]), Path(spec[1]), device)
        loaded.append((cfg, model, Path(spec[2]) if len(spec) == 3 else None))

    base_cfg = loaded[0][0]
    classes = list(base_cfg.data.mvp_classes)
    for cfg, _, _ in loaded[1:]:
        if list(cfg.data.mvp_classes) != classes or cfg.data.image_size != base_cfg.data.image_size:
            raise SystemExit("All models must share the same class list and image_size to be compared on identical images.")
    for c in args.require_class:
        if c not in classes:
            raise SystemExit(f"--require_class {c!r} is not one of {classes}")

    threshold_tensors = []
    for cfg, _, explicit in loaded:
        path = explicit or Path("results") / cfg.name / "thresholds.json"
        th = load_thresholds(path, classes)
        if th is None:
            print(f"[viz] WARNING: no thresholds at {path}; using 0.5 for {cfg.name}. Run scripts/evaluate.py first for tuned ones.")
            th = {c: 0.5 for c in classes}
        else:
            print(f"[viz] {cfg.name}: thresholds from {path}: {th}")
        threshold_tensors.append(torch.tensor([th[c] for c in classes], device=device))

    ds, has_gt, description = build_dataset(args, base_cfg, classes, build_eval_transform(base_cfg.data.image_size))
    if not has_gt:
        for flag, bad in (("--select worst/best", args.select != "random"),
                          ("--require_class", bool(args.require_class)), ("--source", bool(args.source))):
            if bad:
                raise SystemExit(f"{flag} needs ground truth, but --image_dir images have none.")

    candidates = candidate_indices(ds, classes, source=args.source, require_classes=args.require_class) if has_gt else list(range(len(ds)))
    if not candidates:
        raise SystemExit("No images match --source/--require_class.")
    seed = resolve_seed(args.seed)
    print(f"[viz] data: {description}: {len(candidates)} candidate images | random seed {seed} "
          f"({'given' if args.seed is not None else 'fresh — pass --seed ' + str(seed) + ' to reproduce this draw'})")

    def sample_for(index: int):
        item = ds[index]
        image = denormalize(item["image"].numpy(), IMAGENET_MEAN, IMAGENET_STD)
        gt = item["mask"].numpy() > 0.5 if has_gt else None
        return item, image, gt, item["validity_mask"].numpy()[0] > 0.5

    if args.select == "random":
        chosen = pick_random(candidates, args.num_images, seed)
    else:
        pool = pick_random(candidates, args.max_scan, seed)
        cfg0, model0, _ = loaded[0]
        print(f"[viz] ranking {len(pool)} images by {cfg0.name} macro Dice to pick the {args.select} {args.num_images} ...")
        scored = []
        for idx in pool:
            item, _, gt, valid = sample_for(idx)
            pred = predict(model0, item["image"], threshold_tensors[0], device, cfg0.training.amp)
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
        item, image, gt, valid = sample_for(idx)
        preds, scores = {}, {}
        for (cfg, model, _), th in zip(loaded, threshold_tensors):
            preds[cfg.name] = predict(model, item["image"], th, device, cfg.training.amp)
            scores[cfg.name] = macro_dice_over_gt_classes(per_class_dice(preds[cfg.name], gt, valid), gt, valid) if has_gt else float("nan")
        title = f"{item['source_dataset']}  {item['image_id']}"
        fig = render_comparison(image, gt, preds, valid, classes, title, dpi=args.dpi)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item["image_id"]))
        path = args.output_dir / f"{rank:02d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', item['source_dataset'])}_{safe_id}.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
        rows.append((path.name, scores))
        selection.append({"rank": rank, "index": int(idx), "source_dataset": item["source_dataset"], "image_id": str(item["image_id"]), "file": path.name})

    cv2.imwrite(str(args.output_dir / "overview.png"), stack_vertically([cv2.imread(str(w)) for w in written]))
    (args.output_dir / "selection.json").write_text(json.dumps({
        "data": description, "seed": seed, "seed_was_given": args.seed is not None, "select": args.select,
        "models": [{"config": s[0], "checkpoint": s[1]} for s in args.model], "images": selection}, indent=2))

    names = [cfg.name for cfg, _, _ in loaded]
    if has_gt:
        print(f"\n{'image':46s} " + " ".join(f"{n[:22]:>22s}" for n in names))
        for fname, sc in rows:
            print(f"{fname:46s} " + " ".join(f"{sc[n]:22.3f}" for n in names))
    print(f"\n[viz] wrote {len(written)} figures + overview.png + selection.json to {args.output_dir} (seed {seed})")


if __name__ == "__main__":
    main()
