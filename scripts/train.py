"""Train from an experiment YAML.

    python scripts/train.py --config configs/experiments/baseline_smp_efficientnet_b0.yaml
    python scripts/train.py --config <yaml> --wandb_mode disabled --max_epochs 1 --max_train_images 64

The override flags exist for smoke tests; a real run should just use the YAML.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loader import load_experiment_config
from training.trainer import Trainer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--max_train_images", type=int, default=None, help="Smoke test: truncate train set.")
    p.add_argument("--max_val_images", type=int, default=None, help="Smoke test: truncate val set.")
    args = p.parse_args()

    cfg = load_experiment_config(args.config)
    if args.wandb_mode:
        cfg.wandb.mode = args.wandb_mode
    if args.wandb_entity:
        cfg.wandb.entity = args.wandb_entity
    if args.wandb_project:
        cfg.wandb.project = args.wandb_project
    if args.max_epochs:
        cfg.training.max_epochs = args.max_epochs
    if args.batch_size:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.checkpoint_dir:
        cfg.training.checkpoint_dir = args.checkpoint_dir

    trainer = Trainer(cfg, max_train_images=args.max_train_images, max_val_images=args.max_val_images)
    trainer.train()


if __name__ == "__main__":
    main()
