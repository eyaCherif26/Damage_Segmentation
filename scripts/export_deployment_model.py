"""Export a self-contained deployment bundle from an existing (config, checkpoint) pair, so
scripts/predict_deployment.py can run predictions afterward without any experiment YAML at all.

    python scripts/export_deployment_model.py \
        --config configs/experiments/effb0_task_pretrained_cif.yaml \
        --checkpoint runs/effb0_task_pretrained_cif/best.pt \
        --output_dir deploy/effb0_task_pretrained_cif

Writes, into --output_dir:
    model.pt          the COMPLETE model object (torch.save(model, ...), NOT a state_dict) — loadable
                      with a plain torch.load(path, weights_only=False), no configs/models registry,
                      no ExperimentConfig, needed to use it afterward.
    classes.json      [{"id": 1, "name": "Cracks"}, ...] — 1-based ids, in the model's own output-
                      channel order (channel i of the model's prediction is classes[i]).
    meta.json         the handful of facts prediction actually needs (tile size, native_crop scale) —
                      not a copy of the training config (no paths, LRs, augmentation, loss weights, ...).
    thresholds.json   copied from results/<experiment>/thresholds.json if it exists (val-selected,
                      never re-tuned here) — omitted if that file doesn't exist yet.

Caveat, so this isn't oversold: model.pt is a pickled Python object, not a framework-independent
format (not ONNX/TorchScript) — the environment that later loads it still needs this repo's
`models` package importable on PYTHONPATH (the same is true of every "full model" checkpoint format
seen in prior projects). What this DOES remove is the *config* dependency: no YAML, no
configs.loader/configs.schema, no models.build_model() call, no re-deriving the class list or tile
size from a training config at prediction time — just this bundle.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import models
from configs.loader import load_experiment_config
from metrics.evaluation import parse_json_with_helpful_error


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--thresholds", type=Path, default=None,
                   help="thresholds.json to bundle (default: results/<experiment name>/thresholds.json if present)")
    args = p.parse_args()

    cfg = load_experiment_config(args.config)
    model = models.build_model(cfg.model)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.pt"
    torch.save(model, model_path)  # the whole object — see module docstring's portability caveat

    classes = list(cfg.data.mvp_classes)
    (args.output_dir / "classes.json").write_text(json.dumps(
        [{"id": i + 1, "name": c} for i, c in enumerate(classes)], indent=2))

    (args.output_dir / "meta.json").write_text(json.dumps({
        "tile_size": cfg.data.image_size,
        "resize_mode": cfg.data.resize_mode,
        "pixel_scale": cfg.data.pixel_scale,
        "source_config": str(args.config),
        "source_checkpoint": str(args.checkpoint),
        "source_epoch": state.get("epoch"),
    }, indent=2))

    thr_path = args.thresholds or (Path("results") / cfg.name / "thresholds.json")
    if thr_path.exists():
        parse_json_with_helpful_error(thr_path)  # validate before bundling; raises with a clear hint if invalid
        shutil.copyfile(thr_path, args.output_dir / "thresholds.json")
        print(f"[export] bundled thresholds from {thr_path}")
    else:
        print(f"[export] no thresholds file at {thr_path} — bundle has none; "
              "scripts/predict_deployment.py will need --threshold or --thresholds at predict time")

    size_mb = model_path.stat().st_size / 1e6
    has_thr = (args.output_dir / "thresholds.json").exists()
    print(f"[export] wrote {args.output_dir}/ — model.pt ({size_mb:.1f} MB), classes.json, meta.json"
          f"{', thresholds.json' if has_thr else ''}")
    print(f"[export] classes (in the model's own output-channel order): {classes}")


if __name__ == "__main__":
    main()
