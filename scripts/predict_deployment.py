"""Deployment inference: predict on one image using ONLY a bundle from
scripts/export_deployment_model.py — no experiment YAML, no configs/models registry.

    python scripts/predict_deployment.py --bundle deploy/effb0_task_pretrained_cif \
        --image path/to/scene.jpg --output_dir out/scene

Loads `<bundle>/model.pt` (a complete model object, `torch.load(..., weights_only=False)`) and
`<bundle>/classes.json`, tiles the image (sliding-window, same probability-averaging as
scripts/infer_orthophoto.py), applies thresholds, and writes ONE file: `predictions.coco.json`
(COCO-format: images/annotations/categories, one polygon annotation per detected instance, scored
by mean probability — see inference/postprocess.py::export_coco_predictions). Pass --overlay to
also save a quick colour-overlay PNG for a visual sanity check; not needed for the COCO output.

THRESHOLDS, in increasing precedence:
    <bundle>/thresholds.json   used automatically if the bundle has one (val-selected, not re-tuned here)
    --thresholds FILE.json     use this file instead of the bundle's
    --threshold VALUE          ignore both of the above; one fixed value for every class
Neither present -> 0.5 for every class, with a warning.
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

from data.validity_mask import no_data_validity_mask
from inference.postprocess import apply_thresholds, export_coco_predictions
from inference.sliding_window import run_tiled_inference
from metrics.evaluation import parse_json_with_helpful_error, resolve_class_subset


def _repair_transformers_output_capturing(model: torch.nn.Module) -> None:
    """Some HuggingFace model classes (e.g. SegFormer's encoder) capture their internal per-stage
    hidden states via forward hooks that `transformers` installs lazily on first use — gated by a
    process-global registry (`_CAN_RECORD_REGISTRY`) that is only ever populated inside
    `PreTrainedModel.__init__`. A plain `torch.load()` reconstructs the object WITHOUT calling
    `__init__` (pickling restores `__dict__` directly), so that registration never happens for a
    model that only ever went through torch.load in this process — its first real forward pass
    then silently gets `None` where it expected captured hidden states.

    `models.build_model()` (every other script) never hits this: it always constructs the model via
    its real constructor first. This is specific to the pickled-model deployment path, so the fix
    lives here rather than in `models/segformer_model.py`. Best-effort and defensive: if
    `transformers` isn't installed, or this internal API has changed, this quietly does nothing —
    it can only help, never break, a model that doesn't need it (e.g. the smp/torchvision ones)."""
    try:
        from transformers.modeling_utils import PreTrainedModel
        from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY
    except Exception:
        return
    for module in model.modules():
        if isinstance(module, PreTrainedModel):
            key = str(module.__class__)
            if key not in _CAN_RECORD_REGISTRY:
                _CAN_RECORD_REGISTRY[key] = getattr(module, "_can_record_outputs", None)


def load_bundle(bundle_dir: Path, device: torch.device):
    model = torch.load(bundle_dir / "model.pt", map_location=device, weights_only=False)
    model.eval()
    _repair_transformers_output_capturing(model)
    classes = [c["name"] for c in json.loads((bundle_dir / "classes.json").read_text())]
    meta = json.loads((bundle_dir / "meta.json").read_text())
    return model, classes, meta


def resolve_thresholds(bundle_dir: Path, classes: list[str], args) -> dict[str, float]:
    if args.threshold is not None:
        if not (0.0 <= args.threshold <= 1.0):
            raise SystemExit(f"--threshold must be in [0, 1], got {args.threshold}")
        print(f"[predict] fixed threshold {args.threshold} for every class")
        return {c: args.threshold for c in classes}

    path = args.thresholds or (bundle_dir / "thresholds.json")
    if not path.exists():
        if args.thresholds:
            raise FileNotFoundError(f"--thresholds {path} does not exist")
        print(f"[predict] WARNING: no thresholds at {path}; using 0.5 for every class")
        return {c: 0.5 for c in classes}
    raw = parse_json_with_helpful_error(path)
    missing = [c for c in classes if c not in raw]
    if missing:
        raise KeyError(f"{path} has no threshold for class(es) {missing}")
    thresholds = {c: float(raw[c]["threshold"] if isinstance(raw[c], dict) else raw[c]) for c in classes}
    print(f"[predict] thresholds from {path}: {thresholds}")
    return thresholds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, type=Path, help="directory from scripts/export_deployment_model.py")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=None, help="one fixed value for every class, ignoring any thresholds file")
    p.add_argument("--thresholds", type=Path, default=None, help="thresholds.json to use instead of <bundle>/thresholds.json")
    p.add_argument("--tile_size", type=int, default=None, help="default: the bundle's own training tile size")
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Resample the image by this factor before tiling (GSD normalization). "
                        "For a native_crop bundle, match the bundle's own pixel_scale (see meta.json) "
                        "to your image's real-world scale; probability maps are resampled back to "
                        "original pixel coordinates either way.")
    p.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
                   help="Process only this window (original pixels: left, top, width, height).")
    p.add_argument("--min_area", type=int, default=16, help="Drop predicted components smaller than this many px.")
    p.add_argument("--max_memory_gb", type=float, default=8.0)
    p.add_argument("--allow_large_memory", action="store_true")
    p.add_argument("--overlay", action="store_true", help="also write a quick colour-overlay PNG (not needed for the COCO output)")
    p.add_argument("--only_classes", nargs="+", default=None, metavar="CLASS",
                   help="write outputs for only these of the bundle's own classes (not retraining — the model "
                        "still predicts every class it knows)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, all_classes, meta = load_bundle(args.bundle, device)
    classes, channel_indices = resolve_class_subset(all_classes, args.only_classes)  # fail fast on a bad name
    tile_size = args.tile_size or meta["tile_size"]
    print(f"[predict] loaded {args.bundle}/model.pt (classes: {all_classes}, tile_size: {tile_size}, "
          f"trained with resize_mode={meta['resize_mode']}, pixel_scale={meta['pixel_scale']})")
    if args.only_classes:
        print(f"[predict] --only_classes: writing outputs only for {classes} (of the bundle's {all_classes})")

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    full_h, full_w = image.shape[:2]
    if args.crop:
        x, y, cw, ch = args.crop
        if cw <= 0 or ch <= 0 or x < 0 or y < 0 or x + cw > full_w or y + ch > full_h:
            raise SystemExit(f"--crop {args.crop} is outside the image ({full_w}x{full_h}) or empty")
        image = np.ascontiguousarray(image[y : y + ch, x : x + cw])
    h, w = image.shape[:2]
    print(f"[predict] {args.image} ({full_w}x{full_h})" +
          (f", crop {args.crop} -> {w}x{h} (COCO output below is in crop coordinates)" if args.crop else ""))

    model_input = image
    if args.scale != 1.0:
        model_input = cv2.resize(image, (round(w * args.scale), round(h * args.scale)), interpolation=cv2.INTER_AREA)
        print(f"[predict] rescaled x{args.scale} -> {model_input.shape[1]}x{model_input.shape[0]} for the model")

    prob_map = run_tiled_inference(
        model, model_input, len(all_classes), tile_size, args.overlap, args.batch_size, device,
        max_memory_gb=args.max_memory_gb, allow_large_memory=args.allow_large_memory,
    )
    if args.scale != 1.0:
        prob_map = np.stack([cv2.resize(prob_map[i], (w, h), interpolation=cv2.INTER_LINEAR) for i in range(prob_map.shape[0])])
        prob_map *= no_data_validity_mask(image)  # resampling bleeds probability across the no-data edge
    if channel_indices is not None:
        prob_map = prob_map[channel_indices]  # the model still predicted every class; this just prunes what gets written

    thresholds = resolve_thresholds(args.bundle, classes, args)
    binary = apply_thresholds(prob_map, classes, thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_coco_predictions(binary, prob_map, classes, 1, w, h, str(args.output_dir / "predictions.coco.json"), min_area=args.min_area)
    share = {c: round(float(binary[i].mean()), 4) for i, c in enumerate(classes)}
    print(f"[predict] wrote {args.output_dir / 'predictions.coco.json'}; predicted pixel share per class: {share}")

    if args.overlay:
        from inference.visualize import class_colors
        colors = class_colors(len(classes))
        overlay = image.copy()
        for i in range(len(classes)):
            m = binary[i].astype(bool)
            overlay[m] = (0.5 * overlay[m] + 0.5 * colors[i]).astype(np.uint8)
        cv2.imwrite(str(args.output_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"[predict] wrote {args.output_dir / 'overlay.png'}")


if __name__ == "__main__":
    main()
