"""Evaluate a trained checkpoint.

A) The configured data: thresholds selected on val, metrics reported on test (and val).

    python scripts/evaluate.py --config configs/experiments/baseline_smp_efficientnet_b0.yaml \
        --checkpoint runs/baseline_smp_efficientnet_b0/best.pt --output_dir results/baseline_smp_efficientnet_b0

   Writes: results_test.csv, results_val.csv, thresholds.json. Thresholds are ALWAYS chosen on val.

B) ANY other labelled dataset (annotation file in COCO format + its image folder):

    python scripts/evaluate.py --config <config> --checkpoint <best.pt> --output_dir results/<experiment> \
        --coco path/to/annotations.json --images path/to/images [--label_map path/to/map.yaml] [--name my_data]

   Writes results_<name>.csv (name defaults to the annotation file's folder name). The dataset's categories must
   already be MVP class names, or --label_map must map every one of them (null = negative). Metrics use the
   per-class thresholds chosen on the CONFIGURED val split — they are never tuned on the new data, so the numbers
   measure transfer honestly. Thresholds come from --thresholds FILE, else <output_dir>/thresholds.json, else they
   are selected on the configured val split now (and saved there). Classes the dataset has no labels for are
   reported with gt_pixels 0 and left out of the `macro_present` average.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Subset

import models
from configs.loader import load_experiment_config
from data.datasets.combined_dataset import build_split_datasets
from data.datasets.external import build_labelled_dataset
from data.datasets.transforms import build_eval_transform
from metrics.evaluation import (
    evaluate_split,
    load_thresholds,
    results_to_dataframe,
    sanity_check_gt_as_prediction,
    select_thresholds_on_val,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_images", type=int, default=None, help="Smoke test: evaluate on a seeded subset of each split.")
    p.add_argument("--skip_sanity_check", action="store_true")
    ext = p.add_argument_group("evaluate on another labelled dataset instead of the configured val/test")
    ext.add_argument("--coco", type=Path, default=None, help="COCO annotation file of the dataset (needs --images)")
    ext.add_argument("--images", type=Path, default=None, help="image folder belonging to --coco")
    ext.add_argument("--label_map", type=Path, default=None,
                     help="YAML mapping the --coco categories to MVP classes (null = negative); omitted -> categories must already be MVP class names")
    ext.add_argument("--name", default=None, help="label for the dataset in the output file names (default: the annotation file's folder name)")
    ext.add_argument("--thresholds", type=Path, default=None, help="thresholds.json to apply (default: <output_dir>/thresholds.json, else selected on the configured val split)")
    args = p.parse_args()
    if args.coco and not args.images:
        p.error("--coco needs --images (the folder holding that COCO file's images)")
    if (args.images or args.label_map or args.name or args.thresholds) and not args.coco:
        p.error("--images/--label_map/--name/--thresholds only apply together with --coco")

    cfg = load_experiment_config(args.config)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    classes = cfg.data.mvp_classes

    eval_tf = build_eval_transform(cfg.data.image_size)
    batch_size = args.batch_size or cfg.data.batch_size

    def loader(ds, subset_seed_offset: int = 0) -> DataLoader:
        if args.max_images and args.max_images < len(ds):
            g = torch.Generator().manual_seed(cfg.seed + subset_seed_offset)
            ds = Subset(ds, torch.randperm(len(ds), generator=g)[: args.max_images].tolist())
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    def configured_datasets() -> dict:
        # native_crop: thresholds are chosen on a FULL tile grid of the val images (unbiased: every pixel counts,
        # unlike the foreground-biased fixed crops used for per-epoch monitoring), and test is a full tile grid too.
        datasets = build_split_datasets(cfg.data, eval_tf, eval_tf, val_mode="grid")
        if cfg.data.resize_mode == "native_crop":
            print(f"[eval] native_crop: every image is evaluated as non-overlapping {cfg.data.image_size}px tiles at pixel_scale "
                  f"{cfg.data.pixel_scale} — val {len(datasets['val'])} tiles, test {len(datasets['test'])} tiles")
        return datasets

    external = None
    if args.coco:
        ds, info = build_labelled_dataset(args.coco, args.images, cfg.data, eval_tf, args.label_map)
        name = args.name or info["name"]
        tiles = f" -> {len(ds)} tiles at pixel_scale {cfg.data.pixel_scale}" if cfg.data.resize_mode == "native_crop" else ""
        print(f"[eval] external dataset '{name}': {info['n_images']} images{tiles}; categories {info['categories']}")
        if info["unmapped"]:
            print(f"[eval] categories mapped to no MVP class (treated as background): {info['unmapped']}")
        if info["absent_classes"]:
            print(f"[eval] WARNING: this dataset has no labels for {info['absent_classes']} — those classes cannot be scored "
                  "(gt_pixels 0); use the `macro_present` row, not `macro`")
        external = (name, loader(ds, 1))
        val_loader = test_loader = None
    else:
        datasets = configured_datasets()
        val_loader, test_loader = loader(datasets["val"]), loader(datasets["test"])

    if not args.skip_sanity_check:
        sanity_check_gt_as_prediction(external[1] if external else val_loader, classes, device)
        print("[sanity] GT-as-prediction check passed (pixel, boundary, instance paths)")

    model = models.build_model(cfg.model)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    print(f"[eval] loaded {args.checkpoint} (epoch {state['epoch']})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    thr_path = args.thresholds or (args.output_dir / "thresholds.json")
    if external is None or not thr_path.exists():
        if external is not None:
            if args.thresholds:
                raise FileNotFoundError(f"--thresholds {thr_path} does not exist")
            print(f"[eval] {thr_path} not found: selecting thresholds on the configured val split (not on the new data)")
            val_loader = loader(configured_datasets()["val"])
        thresholds = select_thresholds_on_val(model, val_loader, classes, device, amp=cfg.training.amp)
        print("[eval] thresholds chosen on val:", {c: t["threshold"] for c, t in thresholds.items()})
        (args.output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    else:
        flat_loaded = load_thresholds(thr_path, classes)  # raises if a class is missing
        raw = json.loads(thr_path.read_text())
        thresholds = {c: {"threshold": flat_loaded[c],
                          "val_dice": float(raw[c].get("val_dice", float("nan"))) if isinstance(raw[c], dict) else float("nan")}
                      for c in classes}
        print(f"[eval] thresholds from {thr_path} (chosen on val, not tuned on the new data):", flat_loaded)

    flat = {c: t["threshold"] for c, t in thresholds.items()}
    jobs = [(f"external:{external[0]}", external[1], f"results_{external[0]}.csv")] if external else [
        ("val", val_loader, "results_val.csv"), ("test", test_loader, "results_test.csv")]
    for split, ld, fname in jobs:
        result = evaluate_split(model, ld, classes, cfg.model.thin_structure_classes, flat, device, amp=cfg.training.amp)
        df = results_to_dataframe(result, classes, thresholds, split)
        df.to_csv(args.output_dir / fname, index=False)
        overall = df[(df.scope == "overall") & df["class"].isin(["macro", "macro_present", "frequency_weighted"])]
        per_class = df[(df.scope == "overall") & df["class"].isin(classes)]
        print(f"[eval] {split}:\n{overall[['class', 'iou', 'dice', 'precision', 'recall']].to_string(index=False)}")
        if external:
            print(per_class[["class", "dice", "iou", "precision", "recall", "gt_pixels"]].to_string(index=False))
    print(f"[eval] wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
