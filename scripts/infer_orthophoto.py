"""Sliding-window inference on a large orthophoto (JPEG/PNG/TIFF readable by OpenCV).

    python scripts/infer_orthophoto.py --config configs/experiments/baseline_smp_efficientnet_b0.yaml \
        --checkpoint runs/baseline_smp_efficientnet_b0/best.pt --image path/to/ortho.png \
        --output_dir out/ortho --thresholds results/baseline_smp_efficientnet_b0/thresholds.json

Writes per-class probability PNGs (8-bit), a colour overlay (+ a downscaled overlay_preview.jpg), a COCO predictions
file and run_info.json (all settings, for comparing runs). `--crop X Y W H` processes just a window, handy for choosing --scale.
Thresholds default to results/<experiment name>/thresholds.json from scripts/evaluate.py (val-selected), else 0.5 with a warning. Very large images are refused past --max_memory_gb unless --allow_large_memory is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

import models
from configs.loader import load_experiment_config
from inference.postprocess import apply_thresholds, export_coco_predictions
from inference.visualize import class_colors
from metrics.evaluation import load_thresholds, resolve_class_subset
from data.validity_mask import no_data_validity_mask
from inference.sliding_window import run_tiled_inference



def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--thresholds", type=Path, default=None)
    p.add_argument("--tile_size", type=int, default=None, help="Default: data.image_size from the config.")
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_memory_gb", type=float, default=8.0)
    p.add_argument("--allow_large_memory", action="store_true")
    p.add_argument("--min_area", type=int, default=16, help="Drop predicted components smaller than this many px.")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Resample the image by this factor before tiling (GSD normalization, e.g. 0.5 to match "
                        "training images that were downscaled 2x). Probability maps are resampled back, so all "
                        "outputs stay in original pixel coordinates.")
    p.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
                   help="Process only this window (original pixels: left, top, width, height). Use it to try --scale values "
                        "quickly on a small region of a huge orthophoto. COCO output is then in crop coordinates.")
    p.add_argument("--preview_max_side", type=int, default=4096,
                   help="Also write overlay_preview.jpg downscaled to this longest side (a full-size PNG of a huge "
                        "orthophoto is slow to open). 0 disables.")
    p.add_argument("--only_classes", nargs="+", default=None, metavar="CLASS",
                   help="write outputs for only these of the checkpoint's own classes (validated against "
                        "cfg.data.mvp_classes); not retraining, the model still predicts every class it knows")
    args = p.parse_args()

    cfg = load_experiment_config(args.config)
    all_classes = cfg.data.mvp_classes
    classes, channel_indices = resolve_class_subset(all_classes, args.only_classes)  # fail fast on a bad name
    if args.only_classes:
        print(f"[infer] --only_classes: writing outputs only for {classes} (of this checkpoint's {all_classes})")
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    model = models.build_model(cfg.model)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    full_h, full_w = image.shape[:2]
    crop_origin = (0, 0)
    if args.crop:
        x, y, cw, ch = args.crop
        if cw <= 0 or ch <= 0 or x < 0 or y < 0 or x + cw > full_w or y + ch > full_h:
            raise SystemExit(f"--crop {args.crop} is outside the image ({full_w}x{full_h}) or empty")
        image = np.ascontiguousarray(image[y : y + ch, x : x + cw])
        crop_origin = (x, y)
    h, w = image.shape[:2]
    print(f"[infer] {args.image} ({full_w}x{full_h})" + (f", crop {args.crop} -> {w}x{h}" if args.crop else "")
          + f", tile={args.tile_size or cfg.data.image_size}, overlap={args.overlap}")

    model_input = image
    if args.scale != 1.0:
        model_input = cv2.resize(image, (round(w * args.scale), round(h * args.scale)), interpolation=cv2.INTER_AREA)
        print(f"[infer] rescaled x{args.scale} -> {model_input.shape[1]}x{model_input.shape[0]} for the model")

    prob_map = run_tiled_inference(
        model, model_input, len(all_classes), args.tile_size or cfg.data.image_size, args.overlap, args.batch_size, device,
        max_memory_gb=args.max_memory_gb, allow_large_memory=args.allow_large_memory,
    )
    if args.scale != 1.0:
        prob_map = np.stack([cv2.resize(prob_map[i], (w, h), interpolation=cv2.INTER_LINEAR) for i in range(prob_map.shape[0])])
        prob_map *= no_data_validity_mask(image)  # resampling bleeds probability across the no-data edge; re-mask at full resolution
    if channel_indices is not None:
        prob_map = prob_map[channel_indices]  # the model still predicted every class; this just prunes what gets written

    thr_path = args.thresholds or Path("results") / cfg.name / "thresholds.json"
    thresholds = load_thresholds(thr_path, classes)
    if thresholds is None:
        if args.thresholds:
            raise FileNotFoundError(f"--thresholds file not found: {args.thresholds}")
        print(f"[infer] WARNING: no thresholds at {thr_path}; using 0.5 for every class. Run scripts/evaluate.py first for val-tuned ones.")
        thresholds = {c: 0.5 for c in classes}
    else:
        print(f"[infer] thresholds from {thr_path}: {thresholds}")
    binary = apply_thresholds(prob_map, classes, thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = class_colors(len(classes))
    overlay = image.copy()
    for i, name in enumerate(classes):
        cv2.imwrite(str(args.output_dir / f"prob_{name}.png"), (prob_map[i] * 255).astype(np.uint8))
        m = binary[i].astype(bool)
        overlay[m] = (0.5 * overlay[m] + 0.5 * colors[i]).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(args.output_dir / "overlay.png"), overlay_bgr)
    if args.preview_max_side and max(h, w) > args.preview_max_side:
        f = args.preview_max_side / max(h, w)
        cv2.imwrite(str(args.output_dir / "overlay_preview.jpg"), cv2.resize(overlay_bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA))
    export_coco_predictions(binary, prob_map, classes, 1, w, h, str(args.output_dir / "predictions.coco.json"), min_area=args.min_area)
    share = {c: round(float(binary[i].mean()), 4) for i, c in enumerate(classes)}
    (args.output_dir / "run_info.json").write_text(json.dumps({
        "image": str(args.image), "image_size": [full_w, full_h], "crop_xywh": args.crop, "crop_origin": list(crop_origin),
        "scale": args.scale, "tile_size": args.tile_size or cfg.data.image_size, "overlap": args.overlap,
        "config": str(args.config), "checkpoint": str(args.checkpoint), "thresholds": thresholds,
        "class_colors_rgb": {c: [int(v) for v in colors[i]] for i, c in enumerate(classes)},
        "predicted_pixel_share": share, "min_area": args.min_area}, indent=2))
    print(f"[infer] wrote outputs to {args.output_dir}; predicted pixel share per class: {share}")
    print("[infer] overlay colours:", {c: tuple(int(v) for v in colors[i]) for i, c in enumerate(classes)})

if __name__ == "__main__":
    main()
