"""Verify that a run really started from task-pretrained weights (works for any backbone).

    python scripts/check_pretrained_init.py --config configs/experiments/segformer_task_pretrained.yaml \
        [--checkpoint runs/<name>/best.pt]

CHECK 1 (init): builds the same model twice — once ImageNet-only, once task_pretrained — and compares the
backbone. Loading must change (nearly) every backbone tensor; if they are identical the checkpoint had no
effect. Also prints the loader's own report (which file, how many tensors, what was skipped).
CHECK 2 (optional, trained checkpoint): relative L2 distance of the trained backbone to each of the two
initialisations. Early in training it should sit much closer to the task-pretrained one; after long
fine-tuning both distances grow and this comparison stops discriminating — CHECK 1 is the proof.
Runs on CPU, so it is safe while a training job uses the GPU.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import models
from configs.loader import load_experiment_config
from models.init_modes import apply_init_mode


def backbone_named_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Names + values of the backbone only, via the wrapper's own backbone_parameters()."""
    backbone_ids = {id(p) for p in model.backbone_parameters()}
    return {n: p.detach().cpu() for n, p in model.named_parameters() if id(p) in backbone_ids}


def rel_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path, default=None, help="trained best.pt/last.pt to compare against the two inits")
    args = p.parse_args()

    cfg = load_experiment_config(args.config)
    if cfg.model.init_mode != "task_pretrained":
        raise SystemExit(f"{args.config} has model.init_mode={cfg.model.init_mode!r}; nothing to check.")

    imagenet_cfg = copy.deepcopy(cfg.model)
    imagenet_cfg.init_mode, imagenet_cfg.task_pretrained_checkpoint = "from_scratch", None
    m_imagenet = models.build_model(imagenet_cfg)
    m_task = models.build_model(cfg.model)
    report = apply_init_mode(m_task, cfg.model)
    print(f"loader report for {cfg.model.name}:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    p_task, p_img = backbone_named_parameters(m_task), backbone_named_parameters(m_imagenet)
    keys = sorted(p_task)
    changed = [k for k in keys if not torch.equal(p_task[k], p_img[k])]
    print(f"\nCHECK 1 — {len(keys)} backbone tensors, task_pretrained vs ImageNet-only init:")
    print(f"  changed by loading the checkpoint: {len(changed)}/{len(keys)} ({len(changed) / len(keys):.1%})")
    print("  ->", "PASS: the backbone really came from the checkpoint" if len(changed) > 0.9 * len(keys)
          else "FAIL: the checkpoint barely changed the backbone")

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
        trained = {k: v for k, v in state.items() if k in p_task}
        missing = [k for k in keys if k not in trained]
        if missing:
            print(f"\nCHECK 2 skipped: {len(missing)} backbone tensors absent from the checkpoint (different architecture?)")
            return
        d_task = torch.tensor([rel_dist(trained[k], p_task[k]) for k in keys])
        d_img = torch.tensor([rel_dist(trained[k], p_img[k]) for k in keys])
        closer = float((d_task < d_img).float().mean())
        print(f"\nCHECK 2 — trained checkpoint {args.checkpoint} ({len(keys)} tensors), relative L2 distance:")
        print(f"  to task-pretrained init: median {d_task.median():.4f}   mean {d_task.mean():.4f}")
        print(f"  to ImageNet init:        median {d_img.median():.4f}   mean {d_img.mean():.4f}")
        print(f"  tensors closer to the task-pretrained init: {closer:.1%}")
        print("  ->", "PASS: the trained backbone still descends from the task-pretrained weights" if closer > 0.9 else
              "INFO: fine-tuning has moved the backbone beyond the gap between the two inits, so this comparison no "
              "longer discriminates; CHECK 1 and the '[init] task_pretrained load report' log line are the proof")


if __name__ == "__main__":
    main()
