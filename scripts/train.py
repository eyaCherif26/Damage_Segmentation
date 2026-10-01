"""Train from an experiment YAML.

    python scripts/train.py --config configs/experiments/baseline_smp_efficientnet_b0.yaml
    python scripts/train.py --config <yaml> --wandb_mode disabled --max_epochs 1 --max_train_images 64

Most override flags exist for smoke tests; a real run should just use the YAML. `--classes` and
`--lr`/`--lr_head`/`--lr_backbone` are the exceptions — real, supported ways to change training
without a separate YAML. `--lr` sets ONE shared value for both backbone and head (useful for
from_scratch training, where — unlike fine-tuning an already-good task_pretrained/pretrained
state — there's no reason the backbone needs a much lower rate than the head); `--lr_head`/
`--lr_backbone` set them individually and take precedence over `--lr` if both are given. A plain
YAML edit (`training.lr_head`/`training.lr_backbone`) does the same thing and is the more
reproducible option for a real run.
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
    p.add_argument("--classes", nargs="+", default=None, metavar="CLASS",
                   help="Train on only these classes instead of the config's data.mvp_classes (order sets "
                        "channel order). A plain YAML edit (data.mvp_classes + matching model.num_classes) "
                        "does the same thing without this flag. Removing a class still referenced by "
                        "model.thin_structure_classes / loss.per_class_alpha / loss.per_class_gamma in the "
                        "YAML raises — edit those in the YAML too.")
    p.add_argument("--lr", type=float, default=None,
                   help="Sets BOTH training.lr_head and training.lr_backbone to this one value (applied "
                        "before --lr_head/--lr_backbone below, so either of those still overrides it "
                        "individually if given too). The default configs use a 10x lower lr_backbone than "
                        "lr_head, aimed at gently fine-tuning an already-good pretrained/task_pretrained "
                        "encoder — for from_scratch training there's less reason for that gap, so a single "
                        "shared rate is often preferable; this is the CLI shortcut for it.")
    p.add_argument("--lr_head", type=float, default=None)
    p.add_argument("--lr_backbone", type=float, default=None)
    args = p.parse_args()

    cfg = load_experiment_config(args.config)
    if args.classes:
        cfg.data.mvp_classes = list(args.classes)
        cfg.model.num_classes = len(args.classes)
        cfg.__post_init__()  # re-run the length/duplicate/unknown-name cross-checks after mutating in place
    if args.lr is not None:
        cfg.training.lr_head = args.lr
        cfg.training.lr_backbone = args.lr
    if args.lr_head is not None:
        cfg.training.lr_head = args.lr_head
    if args.lr_backbone is not None:
        cfg.training.lr_backbone = args.lr_backbone
    if args.lr is not None or args.lr_head is not None or args.lr_backbone is not None:
        cfg.training.__post_init__()  # re-run weight_decay/min_lr guards (min_lr must stay below both LRs)
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
