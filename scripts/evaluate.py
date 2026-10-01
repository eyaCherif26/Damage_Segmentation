"""Evaluate a trained checkpoint.

A) The configured data: by default, thresholds are selected on val (grid search), metrics reported on test (and val).

    python scripts/evaluate.py --config configs/experiments/baseline_smp_efficientnet_b0.yaml \
        --checkpoint runs/baseline_smp_efficientnet_b0/best.pt --output_dir results/baseline_smp_efficientnet_b0

   Writes: results_test.csv, results_val.csv, thresholds.json.

B) ANY other labelled dataset (annotation file in COCO format + its image folder):

    python scripts/evaluate.py --config <config> --checkpoint <best.pt> --output_dir results/<experiment> \
        --coco path/to/annotations.json --images path/to/images [--label_map path/to/map.yaml] [--name my_data]

   Writes results_<name>.csv (name defaults to the annotation file's folder name). The dataset's categories must
   already be MVP class names, or --label_map must map every one of them (null = negative). Classes the dataset has
   no labels for are reported with gt_pixels 0 and left out of the `macro_present` average.

   For a `data.resize_mode: native_crop` checkpoint, (B) defaults to SLIDING-WINDOW inference per image
   (inference/sliding_window.py::run_tiled_inference — the exact function scripts/infer_orthophoto.py /
   scripts/predict_deployment.py use for real deployment, and which those two scripts apply regardless of
   resize_mode): each image gets ONE stitched full-resolution, overlap-blended probability map, scored against
   its full-resolution ground truth. This matters whenever an external dataset's images are larger than the
   training tile size (data.image_size) — the alternative, treating the image as a grid of independent
   fixed-size crops (what the configured val/test split uses, and what (B) used to always do), hard-cuts a
   defect at tile boundaries with no blending between tiles, which is NOT how the model is used in production
   and can score noticeably differently. Pass --grid_eval to get that old behavior back for a native_crop
   checkpoint (e.g. to compare the two, or to match the configured val/test protocol exactly).

   A `whole_image` checkpoint defaults to a plain resize (matching how it was actually trained), but pass
   --sliding_window to force the same tiled inference on it too — useful to check whether tiled, full-resolution
   inference behaves differently for it as well on an external dataset whose images exceed data.image_size,
   exactly the comparison scripts/infer_orthophoto.py itself doesn't gatekeep on resize_mode either.
   --sliding_window and --grid_eval are mutually exclusive. Neither applies to (A).

THRESHOLDS — three ways to skip/override the grid search, any of which can be combined with --class_threshold:
    --thresholds FILE.json          apply an existing thresholds.json as-is (not re-tuned); for (B) this is also
                                     the implicit default if <output_dir>/thresholds.json already exists.
    --fixed_threshold 0.3           apply this ONE value to every class (e.g. to check what a lower, more
                                     recall-favoring cutoff does) — no grid search at all.
    --class_threshold Rust=0.1 ...  override specific classes on top of whichever base above (or the default
                                     grid search) was used; classes not listed keep the base value.
   A run that used any of the three writes its thresholds to `thresholds_used.json`, NOT `thresholds.json` — the
   val-selected `thresholds.json` from a real grid-search run is never overwritten by a what-if run reusing the
   same --output_dir.

RE-TUNING THRESHOLDS ON (B)'s OWN DATA — for when scores on a new dataset (e.g. a different annotator/site) are
much worse than on the configured test split, and you want to know whether that's a threshold/calibration
mismatch or a genuine capability gap. Both need --coco; mutually exclusive with each other and with --thresholds/
--fixed_threshold above (they're alternative ways to pick a base threshold):
    --retune_on_new_data           Grid-searches thresholds on ALL of --coco, then reports metrics on that SAME
                                    data. This is classic train-on-test leakage — NOT a fair generalization
                                    estimate — use it only as a DIAGNOSTIC: if it comes back much higher than the
                                    frozen-threshold result, the gap was substantially about calibration (e.g. a
                                    different annotator's labeling convention shifts the Dice-optimal cutoff); if
                                    it's still low, the gap is real and re-thresholding won't fix it. Writes
                                    thresholds_selftuned.json / results_<name>_selftuned.csv, never the canonical
                                    files, and prints a loud warning either way.
    --calibration_fraction FRACTION  The FAIR version: splits --coco's images (not tiles — grouped by whole image
                                    even for a native_crop/--sliding_window checkpoint) into a calibration subset
                                    (this fraction, used ONLY to grid-search thresholds) and a held-out reporting
                                    subset (the rest, scored with those frozen thresholds) — the same discipline
                                    as the configured val/test split, just applied to the new data. Split is
                                    seeded off cfg.seed, reproducible. Needs enough images for a meaningful split.

--only_classes CLASS [CLASS ...] — NOT retraining: restricts reporting to a subset of this checkpoint's own
   already-trained classes (validated against cfg.data.mvp_classes). The model still predicts every class it
   was trained on; this just hides the rest from thresholds/results. Composes with the threshold options above
   (they only need to cover the filtered subset) and with the "don't overwrite the canonical file" protection.

--detection_iou_threshold IOU (default 0.5) — how much a predicted connected component must overlap a GT instance
   to count as "detected" for the output CSV's inst_recall_{small,medium,large} columns. The default is a strict
   shape/size match: for a thin, curved structure like a crack, a prediction a human would call "found it" often
   falls well under 0.5 IoU, which is why inst_recall can read exactly 0 even when pixel Dice/recall are modestly
   positive and the overlay looks reasonable by eye. Lower this (e.g. 0.1) for a "did it flag the defect at all"
   number — report it ALONGSIDE the strict pixel/boundary metrics (unaffected by this flag), not instead of them;
   it answers a different question (presence vs. precise extent).
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
    evaluate_external_with_tiled_inference,
    evaluate_split,
    load_thresholds,
    parse_threshold_overrides,
    resolve_class_subset,
    results_to_dataframe,
    sanity_check_gt_as_prediction,
    select_thresholds_on_val,
    select_thresholds_with_tiled_inference,
    split_images_for_calibration,
)


def _dataset_image_ids(dataset) -> list:
    """Every item's image id, for --calibration_fraction's by-image split of a non-tiled external
    dataset (a ConcatDataset of one CocoMultiLabelSegDataset (whole_image) or CocoCropDataset
    (native_crop, --grid_eval) — the latter has several tiles per image, hence sample_image_id()
    rather than assuming one item == one image)."""
    subs = dataset.datasets if hasattr(dataset, "datasets") else [dataset]
    ids = []
    for sub in subs:
        for local in range(len(sub)):
            ids.append(sub.sample_image_id(local) if hasattr(sub, "sample_image_id") else sub.image_ids[local])
    return ids


def _flat_indices_for_images(dataset, wanted_ids) -> list[int]:
    """Flat item indices (matching how DataLoader/Subset already index `dataset` elsewhere in this
    script) whose image id is in `wanted_ids` — the other half of _dataset_image_ids's split."""
    wanted = set(wanted_ids)
    subs = dataset.datasets if hasattr(dataset, "datasets") else [dataset]
    out, offset = [], 0
    for sub in subs:
        for local in range(len(sub)):
            image_id = sub.sample_image_id(local) if hasattr(sub, "sample_image_id") else sub.image_ids[local]
            if image_id in wanted:
                out.append(offset + local)
        offset += len(sub)
    return out


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
    ext.add_argument("--sliding_window", action="store_true",
                     help="Force sliding-window, overlap-blended inference (inference/sliding_window.py::run_tiled_inference "
                          "— the exact function scripts/infer_orthophoto.py/predict_deployment.py use for real deployment: "
                          "ONE stitched full-resolution probability map per image) even for a whole_image checkpoint. "
                          "infer_orthophoto.py itself applies this regardless of resize_mode, so this makes evaluate.py "
                          "capable of the same — useful whenever the dataset's images are larger than data.image_size, "
                          "which is exactly where a plain whole-image resize can behave differently than real deployment "
                          "inference does. Already the default for a native_crop checkpoint (see --grid_eval below); this "
                          "flag only changes anything for a whole_image one.")
    ext.add_argument("--grid_eval", action="store_true",
                     help="native_crop only: use the old non-overlapping fixed-tile-grid evaluation (matches the "
                          "configured val/test protocol) instead of the sliding-window overlap-blended inference "
                          "used by default — mainly for comparing the two. Mutually exclusive with --sliding_window.")
    ext.add_argument("--tile_overlap", type=int, default=128,
                     help="sliding-window inference only: tile overlap in pixels (default matches infer_orthophoto.py)")
    ext.add_argument("--max_memory_gb", type=float, default=8.0, help="sliding-window inference only: VRAM budget guard")
    ext.add_argument("--allow_large_memory", action="store_true")
    thr = p.add_argument_group("threshold overrides (skip/override the val grid search; see module docstring)")
    thr.add_argument("--thresholds", type=Path, default=None,
                     help="apply this thresholds.json as-is, not re-tuned (for --coco, default if <output_dir>/thresholds.json exists)")
    thr.add_argument("--fixed_threshold", type=float, default=None, help="apply this one value to every class instead of grid search")
    thr.add_argument("--class_threshold", nargs="+", default=[], metavar="CLASS=VALUE",
                     help="override specific classes' threshold on top of the base method, e.g. --class_threshold Rust=0.1 Efflorescence=0.3")
    retune = p.add_argument_group("re-tune thresholds on --coco's own data instead of the configured val split "
                                  "(read the RE-TUNING section of the module docstring first — is it fair for your case?)")
    retune.add_argument("--retune_on_new_data", action="store_true",
                        help="DIAGNOSTIC, not a fair generalization estimate: grid-search on ALL of --coco, report on "
                             "that same data (classic train-on-test leakage) — use only to check whether a low score "
                             "elsewhere is a threshold/calibration mismatch rather than a genuine capability gap.")
    retune.add_argument("--calibration_fraction", type=float, default=None, metavar="FRACTION",
                        help="FAIR version: splits --coco (by whole image) into a calibration subset (this fraction, "
                             "thresholds only) and a held-out reporting subset (the rest, scored with those frozen "
                             "thresholds) — same discipline as the configured val/test split, applied to the new data.")
    p.add_argument("--only_classes", nargs="+", default=None, metavar="CLASS",
                   help="report only these of the checkpoint's own classes (validated against cfg.data.mvp_classes); "
                        "not retraining, the model still predicts every class it knows")
    p.add_argument("--detection_iou_threshold", type=float, default=0.5, metavar="IOU",
                   help="How much a predicted connected component must overlap a GT instance (IoU) to count it as "
                        "'detected' for inst_recall_{small,medium,large} in the output CSV. Default (0.5) is a strict "
                        "shape/size match — for a thin, curved structure like a crack, a prediction a human would call "
                        "'found it' routinely falls well under 0.5 IoU, which is why inst_recall can read exactly 0 "
                        "even when pixel Dice/recall are modestly positive and the overlay looks reasonable by eye. "
                        "Lower this (e.g. 0.1) for a client-facing 'did it flag the defect at all' number — report it "
                        "ALONGSIDE the strict pixel/boundary metrics (unaffected by this flag), not instead of them; "
                        "it answers a different question (presence vs. precise extent).")
    args = p.parse_args()
    if not (0.0 < args.detection_iou_threshold <= 1.0):
        p.error(f"--detection_iou_threshold must be in (0, 1], got {args.detection_iou_threshold}")
    if args.coco and not args.images:
        p.error("--coco needs --images (the folder holding that COCO file's images)")
    if (args.images or args.label_map or args.name) and not args.coco:
        p.error("--images/--label_map/--name only apply together with --coco")
    if args.fixed_threshold is not None and args.thresholds is not None:
        p.error("--fixed_threshold and --thresholds are mutually exclusive — pick one base method")
    if args.fixed_threshold is not None and not (0.0 <= args.fixed_threshold <= 1.0):
        p.error(f"--fixed_threshold must be in [0, 1], got {args.fixed_threshold}")
    if args.sliding_window and args.grid_eval:
        p.error("--sliding_window and --grid_eval are mutually exclusive — pick one")
    if (args.retune_on_new_data or args.calibration_fraction is not None) and not args.coco:
        p.error("--retune_on_new_data/--calibration_fraction only apply together with --coco")
    if args.retune_on_new_data and args.calibration_fraction is not None:
        p.error("--retune_on_new_data and --calibration_fraction are mutually exclusive — pick one")
    if args.calibration_fraction is not None and not (0.0 < args.calibration_fraction < 1.0):
        p.error(f"--calibration_fraction must be in (0, 1), got {args.calibration_fraction}")
    if (args.retune_on_new_data or args.calibration_fraction is not None) and (args.thresholds is not None or args.fixed_threshold is not None):
        p.error("--retune_on_new_data/--calibration_fraction and --thresholds/--fixed_threshold are different ways "
                "to pick a base threshold — pick one")

    cfg = load_experiment_config(args.config)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    all_classes = cfg.data.mvp_classes
    classes, channel_indices = resolve_class_subset(all_classes, args.only_classes)  # fail fast on a bad name
    if args.only_classes:
        print(f"[eval] --only_classes: reporting only {classes} (of this checkpoint's {all_classes})")

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
    external_tiled_ds = None  # set whenever sliding-window inference is used: see evaluate_external_with_tiled_inference
    if args.coco:
        ds, info = build_labelled_dataset(args.coco, args.images, cfg.data, eval_tf, args.label_map)
        name = args.name or info["name"]
        # native_crop defaults to tiled (opt out with --grid_eval); whole_image defaults to a plain
        # resize but can be forced into it too with --sliding_window — infer_orthophoto.py itself
        # applies run_tiled_inference regardless of resize_mode, so evaluate.py can match it either way.
        use_tiled = (args.sliding_window or cfg.data.resize_mode == "native_crop") and not args.grid_eval
        if args.grid_eval and cfg.data.resize_mode != "native_crop":
            print("[eval] WARNING: --grid_eval has no effect here — this checkpoint is whole_image and "
                  "wasn't going to use sliding-window inference anyway (pass --sliding_window to force it on instead)")
        if use_tiled:
            external_tiled_ds, _ = build_labelled_dataset(args.coco, args.images, cfg.data, None, args.label_map, force_whole_image=True)
            forced = " — forced via --sliding_window on a whole_image checkpoint" if cfg.data.resize_mode != "native_crop" else ""
            print(f"[eval] external dataset '{name}': {info['n_images']} images -> sliding-window inference "
                  f"({cfg.data.image_size}px tiles, {args.tile_overlap}px overlap, pixel_scale {cfg.data.pixel_scale}, "
                  f"blended like scripts/infer_orthophoto.py{forced} — pass --grid_eval for the old fixed-tile-grid behavior); "
                  f"categories {info['categories']}")
        else:
            tiles = f" -> {len(ds)} tiles at pixel_scale {cfg.data.pixel_scale}" if cfg.data.resize_mode == "native_crop" else ""
            print(f"[eval] external dataset '{name}': {info['n_images']} images{tiles}; categories {info['categories']}")
        if info["unmapped"]:
            print(f"[eval] categories mapped to no MVP class (treated as background): {info['unmapped']}")
        if info["absent_classes"]:
            print(f"[eval] WARNING: this dataset has no labels for {info['absent_classes']} — those classes cannot be scored "
                  "(gt_pixels 0); use the `macro_present` row, not `macro`")
        external = (name, loader(ds, 1))  # always built: drives the sanity check either way, and real scoring when not tiled
        val_loader = test_loader = None
    else:
        datasets = configured_datasets()
        val_loader, test_loader = loader(datasets["val"]), loader(datasets["test"])

    if not args.skip_sanity_check:
        sanity_check_gt_as_prediction(external[1] if external else val_loader, all_classes, device)
        print("[sanity] GT-as-prediction check passed (pixel, boundary, instance paths)")

    model = models.build_model(cfg.model)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    print(f"[eval] loaded {args.checkpoint} (epoch {state['epoch']})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_overrides = parse_threshold_overrides(args.class_threshold, classes) if args.class_threshold else {}
    freshly_searched = False

    report_image_ids = None  # set only by --calibration_fraction, to restrict the scoring pass below to the held-out split
    retuned_method = None  # "selftuned" | "calibrated" | None — distinct output-file naming, see below
    if args.fixed_threshold is not None:
        thresholds = {c: {"threshold": args.fixed_threshold, "val_dice": float("nan")} for c in classes}
        base_desc = f"fixed {args.fixed_threshold} for every class"
    else:
        # For --coco, an existing <output_dir>/thresholds.json is the implicit default (no --thresholds needed) —
        # unless --retune_on_new_data/--calibration_fraction explicitly asked for something else instead.
        retuning = args.retune_on_new_data or args.calibration_fraction is not None
        thr_path = args.thresholds or (args.output_dir / "thresholds.json")
        use_file = args.thresholds is not None or (external is not None and thr_path.exists() and not retuning)
        if use_file:
            if not thr_path.exists():
                raise FileNotFoundError(f"--thresholds {thr_path} does not exist")
            flat_loaded = load_thresholds(thr_path, classes)  # raises if a class is missing
            raw = json.loads(thr_path.read_text())
            thresholds = {c: {"threshold": flat_loaded[c],
                              "val_dice": float(raw[c].get("val_dice", float("nan"))) if isinstance(raw[c], dict) else float("nan")}
                          for c in classes}
            base_desc = f"{thr_path} (not re-tuned)"
        elif args.retune_on_new_data:
            print("[eval] WARNING: --retune_on_new_data grid-searches thresholds on ALL of the external dataset, "
                  "then reports metrics on that SAME data — NOT a fair generalization estimate (deliberate "
                  "train-on-test leakage). Use only to check whether a low score elsewhere is a calibration "
                  "mismatch, not a genuine capability gap; see --calibration_fraction for the fair version.")
            if external_tiled_ds is not None:
                thresholds = select_thresholds_with_tiled_inference(
                    model, external_tiled_ds, classes, device, cfg.data.image_size, overlap=args.tile_overlap,
                    batch_size=batch_size, max_memory_gb=args.max_memory_gb, allow_large_memory=args.allow_large_memory,
                    channel_indices=channel_indices,
                )
            else:
                thresholds = select_thresholds_on_val(model, external[1], classes, device, amp=cfg.training.amp,
                                                      channel_indices=channel_indices)
            base_desc, retuned_method = f"grid search on ALL of external:{external[0]} (self-tuned, NOT held-out)", "selftuned"
        elif args.calibration_fraction is not None:
            all_ids = external_tiled_ds.image_ids if external_tiled_ds is not None else _dataset_image_ids(ds)
            calib_ids, report_ids = split_images_for_calibration(all_ids, args.calibration_fraction, cfg.seed)
            print(f"[eval] --calibration_fraction {args.calibration_fraction}: {len(calib_ids)} images for "
                  f"threshold tuning, {len(report_ids)} held out for reporting (seed {cfg.seed})")
            if external_tiled_ds is not None:
                thresholds = select_thresholds_with_tiled_inference(
                    model, external_tiled_ds, classes, device, cfg.data.image_size, overlap=args.tile_overlap,
                    batch_size=batch_size, max_memory_gb=args.max_memory_gb, allow_large_memory=args.allow_large_memory,
                    channel_indices=channel_indices, image_ids=list(calib_ids),
                )
                report_image_ids = list(report_ids)
            else:
                calib_loader = DataLoader(Subset(ds, _flat_indices_for_images(ds, calib_ids)), batch_size=batch_size,
                                          shuffle=False, num_workers=args.num_workers, pin_memory=True)
                thresholds = select_thresholds_on_val(model, calib_loader, classes, device, amp=cfg.training.amp,
                                                      channel_indices=channel_indices)
                external = (external[0], DataLoader(Subset(ds, _flat_indices_for_images(ds, report_ids)), batch_size=batch_size,
                                                    shuffle=False, num_workers=args.num_workers, pin_memory=True))
            base_desc = f"grid search on a {args.calibration_fraction:.0%} calibration split of external:{external[0]} (held-out reporting)"
            retuned_method = "calibrated"
        else:
            if external is not None:
                print("[eval] no thresholds given/found: selecting on the configured val split (not on the new data)")
                val_loader = loader(configured_datasets()["val"])
            thresholds = select_thresholds_on_val(model, val_loader, classes, device, amp=cfg.training.amp,
                                                  channel_indices=channel_indices)
            base_desc, freshly_searched = "grid search on val", True

    if class_overrides:
        for c, v in class_overrides.items():
            thresholds[c] = {"threshold": v, "val_dice": float("nan")}
        base_desc += f"; per-class overrides {class_overrides}"

    print(f"[eval] thresholds ({base_desc}):", {c: t["threshold"] for c, t in thresholds.items()})
    partial = bool(args.only_classes)
    if freshly_searched and not class_overrides and not partial:
        out_name = "thresholds.json"
    elif retuned_method is not None:
        # Its own unmistakable name — never thresholds.json (not val-selected) or thresholds_used.json
        # (that means "an explicitly supplied file was used", a different thing worth not conflating).
        out_name = f"thresholds_{retuned_method}.json"
    elif args.fixed_threshold is not None or class_overrides or args.thresholds is not None or partial:
        # An explicit ask for something other than the default (fixed/override/an explicit --thresholds
        # file/--only_classes) is worth recording; silently re-reading the same default file verbatim is not.
        out_name = "thresholds_used.json"
    else:
        out_name = None
    if out_name is not None:
        (args.output_dir / out_name).write_text(json.dumps(thresholds, indent=2))
        if out_name != "thresholds.json":
            print(f"[eval] wrote {out_name} — any existing thresholds.json (the val-selected record) is left untouched")

    flat = {c: t["threshold"] for c, t in thresholds.items()}
    # Same protection as thresholds.json/thresholds_used.json above: a what-if run (file/fixed/override
    # thresholds, or --only_classes) writes results_{val,test}_used.csv rather than overwriting a real
    # grid-search run's canonical results_{val,test}.csv in the same --output_dir. For external, a
    # retune_method result also gets its own suffix — --retune_on_new_data's number in particular must
    # never be filed under the same name as a real frozen-threshold evaluation.
    result_suffix = "" if (freshly_searched and not class_overrides and not partial) else "_used"
    external_suffix = f"_{retuned_method}" if retuned_method is not None else ""
    jobs = [(f"external:{external[0]}", external[1], f"results_{external[0]}{external_suffix}.csv")] if external else [
        ("val", val_loader, f"results_val{result_suffix}.csv"), ("test", test_loader, f"results_test{result_suffix}.csv")]
    for split, ld, fname in jobs:
        if external_tiled_ds is not None:
            result = evaluate_external_with_tiled_inference(
                model, external_tiled_ds, classes, cfg.model.thin_structure_classes, flat, device,
                tile_size=cfg.data.image_size, overlap=args.tile_overlap, batch_size=batch_size,
                max_memory_gb=args.max_memory_gb, allow_large_memory=args.allow_large_memory,
                channel_indices=channel_indices, image_ids=report_image_ids,
                instance_iou_threshold=args.detection_iou_threshold,
            )
        else:
            result = evaluate_split(model, ld, classes, cfg.model.thin_structure_classes, flat, device, amp=cfg.training.amp,
                                    channel_indices=channel_indices, instance_iou_threshold=args.detection_iou_threshold)
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
