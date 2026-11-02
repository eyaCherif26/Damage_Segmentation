"""Standalone, opt-in rule-based cleanup of an existing infer_orthophoto.py / predict_deployment.py
run's saved predictions: morphological cleanup, hysteresis thresholding, and shape-based (crack)
filtering (see inference/noise_filtering.py), each independently testable on a saved example
without re-running the model.

    python scripts/postprocess_predictions.py --input_dir out/ortho \\
        --hysteresis_low_threshold Cracks=0.1

Reads --input_dir's prob_<class>.png files and run_info.json (class list, per-class thresholds) --
no config/checkpoint/GPU needed. Writes predictions.coco.json + overlay.png + postprocess_info.json
into --output_dir (default: <input_dir>/postprocessed/), never modifying --input_dir's own files.

Each of the three techniques is a no-op unless its own flags are given, so they can be tried one at
a time: run once with only --morph_classes/--morph_close_px set, inspect the result, then try
--hysteresis_low_threshold alone, etc. -- all on the SAME saved probability maps, with no
re-inference between attempts. Combine them once each is understood individually.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from inference.noise_filtering import clean_binary_mask
from inference.postprocess import apply_thresholds, export_coco_predictions
from metrics.evaluation import parse_threshold_overrides


def _load_run(input_dir: Path) -> tuple[np.ndarray, list[str], dict[str, float], dict]:
    """Rebuilds (prob_map, classes, thresholds, run_info) from a prior infer_orthophoto.py/
    predict_deployment.py --output_dir. prob_<class>.png is an 8-bit quantization of the float
    probability (~0.4% precision) -- ample for threshold work at the granularity these techniques
    operate at, so no change to either inference script is needed to support this."""
    run_info = json.loads((input_dir / "run_info.json").read_text())
    thresholds = run_info["thresholds"]
    classes = list(thresholds.keys())
    prob_map = np.stack([
        cv2.imread(str(input_dir / f"prob_{name}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        for name in classes
    ])
    return prob_map, classes, thresholds, run_info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True, type=Path,
                   help="an existing infer_orthophoto.py/predict_deployment.py --output_dir")
    p.add_argument("--output_dir", type=Path, default=None, help="default: <input_dir>/postprocessed/")
    p.add_argument("--min_area", type=int, default=16,
                   help="passed through to export_coco_predictions, same meaning as the original run's --min_area")
    p.add_argument("--segmentation_format", choices=["rle", "polygon"], default="polygon",
                   help="default polygon (not rle), matching infer_orthophoto_batch.sh's own default -- "
                        "so post-processed output has the same hull/segmentation shape as the original "
                        "run's predictions.coco.json unless you explicitly ask for rle here")

    morph = p.add_argument_group("morphological cleanup")
    morph.add_argument("--morph_classes", nargs="+", default=None, metavar="CLASS")
    morph.add_argument("--morph_close_px", type=int, default=0, metavar="PX")
    morph.add_argument("--morph_open_px", type=int, default=0, metavar="PX")

    hyst = p.add_argument_group("hysteresis thresholding")
    hyst.add_argument("--hysteresis_low_threshold", nargs="+", default=[], metavar="CLASS=VALUE",
                       help="low 'candidate' threshold per class; the run's own recorded threshold is "
                            "used as the high/seed value, e.g. --hysteresis_low_threshold Cracks=0.1")

    merge = p.add_argument_group("merge nearby fragments (distance-based)")
    merge.add_argument("--merge_classes", nargs="+", default=None, metavar="CLASS",
                        help="e.g. a crack split into several separate pieces by noisy thresholding -- "
                             "bridges the two nearest skeleton endpoints of two different components "
                             "of this class if they're within --merge_gap_px of each other")
    merge.add_argument("--merge_gap_px", type=int, default=0, metavar="PX",
                        help="REQUIRED (>0) with --merge_classes -- the max endpoint-to-endpoint "
                             "distance that still counts as 'the same crack, briefly missed'")
    merge.add_argument("--merge_spur_prune_px", type=int, default=0, metavar="PX",
                        help="trims short spurious skeleton end-branches before looking for endpoints, "
                             "so a noise whisker's tip isn't mistaken for a real fragment end")

    shape = p.add_argument_group("shape-based (crack) filtering")
    shape.add_argument("--shape_filter_classes", nargs="+", default=None, metavar="CLASS",
                        help="recommended: Cracks only -- a blob class like Spalling would be wrongly "
                             "rejected by a max-width filter")
    shape.add_argument("--shape_filter_max_width_px", type=float, default=None, metavar="PX",
                        help="REQUIRED with --shape_filter_classes, no default on purpose: 'too wide "
                             "to be a crack' is a GSD/domain-dependent calibration choice")
    shape.add_argument("--shape_filter_spur_prune_px", type=int, default=0, metavar="PX")
    shape.add_argument("--shape_filter_link_gap_px", type=int, default=0, metavar="PX")
    args = p.parse_args()

    if (args.morph_close_px or args.morph_open_px) and not args.morph_classes:
        p.error("--morph_close_px/--morph_open_px only apply together with --morph_classes")
    if args.morph_classes and not (args.morph_close_px or args.morph_open_px):
        p.error("--morph_classes given but --morph_close_px and --morph_open_px are both 0 -- would be a no-op")
    if args.shape_filter_max_width_px is not None and not args.shape_filter_classes:
        p.error("--shape_filter_max_width_px only applies together with --shape_filter_classes")
    if args.shape_filter_classes and args.shape_filter_max_width_px is None:
        p.error("--shape_filter_classes requires --shape_filter_max_width_px -- no default on purpose, "
                "'too wide to be a crack' must come from real calibration, not a guess")
    if (args.shape_filter_spur_prune_px or args.shape_filter_link_gap_px) and not args.shape_filter_classes:
        p.error("--shape_filter_spur_prune_px/--shape_filter_link_gap_px only apply together with --shape_filter_classes")
    if args.merge_gap_px and not args.merge_classes:
        p.error("--merge_gap_px only applies together with --merge_classes")
    if args.merge_classes and not args.merge_gap_px:
        p.error("--merge_classes requires --merge_gap_px > 0 -- no default on purpose, 'close enough to be "
                "the same crack' is a GSD/domain-dependent calibration choice")
    if args.merge_spur_prune_px and not args.merge_classes:
        p.error("--merge_spur_prune_px only applies together with --merge_classes")
    if not (args.morph_classes or args.hysteresis_low_threshold or args.merge_classes or args.shape_filter_classes):
        p.error("nothing to do; pass at least one of --morph_classes/--hysteresis_low_threshold/"
                "--merge_classes/--shape_filter_classes")

    prob_map, classes, thresholds, run_info = _load_run(args.input_dir)

    if args.morph_classes:
        unknown = [c for c in args.morph_classes if c not in classes]
        if unknown:
            p.error(f"--morph_classes: {unknown} not in this run's classes {classes}")
    if args.merge_classes:
        unknown = [c for c in args.merge_classes if c not in classes]
        if unknown:
            p.error(f"--merge_classes: {unknown} not in this run's classes {classes}")
    if args.shape_filter_classes:
        unknown = [c for c in args.shape_filter_classes if c not in classes]
        if unknown:
            p.error(f"--shape_filter_classes: {unknown} not in this run's classes {classes}")

    hysteresis_low_thresholds = (
        parse_threshold_overrides(args.hysteresis_low_threshold, classes) if args.hysteresis_low_threshold else {}
    )
    for name, low in hysteresis_low_thresholds.items():
        if low > thresholds[name]:
            p.error(f"--hysteresis_low_threshold {name}={low} must be <= this run's own threshold for "
                    f"{name} ({thresholds[name]})")

    binary = apply_thresholds(prob_map, classes, thresholds)
    cleaned = clean_binary_mask(
        binary, prob_map, classes, thresholds,
        morph_classes=args.morph_classes, morph_close_px=args.morph_close_px, morph_open_px=args.morph_open_px,
        hysteresis_low_thresholds=hysteresis_low_thresholds,
        merge_classes=args.merge_classes, merge_gap_px=args.merge_gap_px, merge_spur_prune_px=args.merge_spur_prune_px,
        shape_filter_classes=args.shape_filter_classes, shape_filter_max_width_px=args.shape_filter_max_width_px,
        shape_filter_spur_prune_px=args.shape_filter_spur_prune_px, shape_filter_link_gap_px=args.shape_filter_link_gap_px,
    )

    output_dir = args.output_dir or (args.input_dir / "postprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w = prob_map.shape[1:]

    export_coco_predictions(
        cleaned, prob_map, classes, 1, w, h, str(output_dir / "predictions.coco.json"),
        min_area=args.min_area, file_name=run_info.get("image"), segmentation_format=args.segmentation_format,
    )

    # Reuse the original orthophoto for the overlay when it's still at the path run_info recorded
    # and its size still matches prob_map's (it won't if the original run used --crop/--scale);
    # otherwise fall back to a plain background -- this overlay is for before/after mask-shape
    # comparison, not a replacement for the original run's own overlay.png.
    base = None
    orig_image = run_info.get("image")
    if orig_image and Path(orig_image).is_file():
        bgr = cv2.imread(orig_image, cv2.IMREAD_COLOR)
        if bgr is not None and bgr.shape[:2] == (h, w):
            base = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if base is None:
        base = np.full((h, w, 3), 40, dtype=np.uint8)

    colors = run_info.get("class_colors_rgb", {})
    overlay = base.copy()
    for i, name in enumerate(classes):
        color = np.array(colors.get(name, [255, 0, 0]), dtype=np.uint8)
        m = cleaned[i].astype(bool)
        overlay[m] = (0.5 * overlay[m] + 0.5 * color).astype(np.uint8)
    cv2.imwrite(str(output_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    before_share = {c: round(float(binary[i].mean()), 4) for i, c in enumerate(classes)}
    after_share = {c: round(float(cleaned[i].mean()), 4) for i, c in enumerate(classes)}
    (output_dir / "postprocess_info.json").write_text(json.dumps({
        "input_dir": str(args.input_dir),
        "morph_classes": args.morph_classes, "morph_close_px": args.morph_close_px, "morph_open_px": args.morph_open_px,
        "hysteresis_low_thresholds": hysteresis_low_thresholds,
        "merge_classes": args.merge_classes, "merge_gap_px": args.merge_gap_px, "merge_spur_prune_px": args.merge_spur_prune_px,
        "shape_filter_classes": args.shape_filter_classes, "shape_filter_max_width_px": args.shape_filter_max_width_px,
        "shape_filter_spur_prune_px": args.shape_filter_spur_prune_px, "shape_filter_link_gap_px": args.shape_filter_link_gap_px,
        "min_area": args.min_area,
        "predicted_pixel_share_before": before_share, "predicted_pixel_share_after": after_share,
    }, indent=2))
    print(f"[postprocess] wrote outputs to {output_dir}")
    print(f"[postprocess] predicted pixel share before -> after: {before_share} -> {after_share}")


if __name__ == "__main__":
    main()
