"""Training loop.

Anti-overfitting mechanisms are concrete, not just principles (see CLAUDE.md for the full
rationale trail): AdamW with non-zero weight decay; early stopping on val macro Dice with
`max_epochs` as a ceiling; both backbone/head param groups built once at optimizer construction
(never `optimizer.add_param_group()` mid-training) with freezing done via `requires_grad`
toggling only; a schema-validation pass over every source dataset before the first epoch; val
data is never augmented; per-epoch train+val curves and per-class tables logged to W&B every
epoch, not just at the end.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

import models
from configs.loader import config_to_dict
from configs.schema import ExperimentConfig
from data.datasets.combined_dataset import build_split_datasets
from data.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform, build_train_transform
from data.schema_validator import validate_coco_file
from losses.combined import CombinedLoss
from metrics.pixel_metrics import PixelMetricAccumulator
from models.init_modes import apply_init_mode
from training.early_stopping import EarlyStopping
from training.schedule import build_warmup_cosine_scheduler
from training.wandb_logger import WandbLogger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        max_train_images: int | None = None,
        max_val_images: int | None = None,
    ) -> None:
        """`max_*_images` are for smoke tests only: they take a seeded random subset (not the first
        N indices, which would be a single source since datasets are concatenated per source)."""
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

        self._validate_datasets()

        train_tf = build_train_transform(cfg.augmentation, cfg.data.image_size)
        eval_tf = build_eval_transform(cfg.data.image_size)
        self.datasets = build_split_datasets(cfg.data, train_tf, eval_tf)
        for split, limit in (("train", max_train_images), ("val", max_val_images)):
            if limit is not None and limit < len(self.datasets[split]):
                idx = np.random.default_rng(cfg.seed).permutation(len(self.datasets[split]))[:limit].tolist()
                self.datasets[split] = Subset(self.datasets[split], idx)

        d = cfg.data
        if d.resize_mode == "native_crop":
            print(f"[data] native_crop: pixel_scale {d.pixel_scale} (jitter {d.train_scale_jitter}), {d.image_size}px crops, "
                  f"fg_crop_prob {d.fg_crop_prob} | per epoch: train {len(self.datasets['train'])} crops "
                  f"({d.train_crops_per_image}/image), val {len(self.datasets['val'])} fixed crops")
        else:
            print(f"[data] whole_image: every image resized to {d.image_size}x{d.image_size} | train {len(self.datasets['train'])}, val {len(self.datasets['val'])} images")
        self.train_loader = DataLoader(
            self.datasets["train"], batch_size=cfg.data.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, drop_last=True, pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.datasets["val"], batch_size=cfg.data.batch_size, shuffle=False,
            num_workers=cfg.data.num_workers, pin_memory=True,
        )

        self.model = models.build_model(cfg.model)
        if cfg.model.init_mode == "task_pretrained":
            init_report = apply_init_mode(self.model, cfg.model)
            print(f"[init] task_pretrained load report: {init_report}")
        self.model.to(self.device)

        self.loss_fn = CombinedLoss(cfg.loss, cfg.data.mvp_classes, cfg.model.thin_structure_classes).to(self.device)

        backbone_params = list(self.model.backbone_parameters())
        head_params = list(self.model.head_parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.training.lr_backbone, "name": "backbone"},
                {"params": head_params, "lr": cfg.training.lr_head, "name": "head"},
            ],
            weight_decay=cfg.training.weight_decay,
        )
        steps_per_epoch = max(1, len(self.train_loader) // cfg.training.grad_accum_steps)
        self.scheduler = build_warmup_cosine_scheduler(self.optimizer, cfg.training.warmup_epochs, cfg.training.max_epochs, steps_per_epoch)

        self.early_stopping = EarlyStopping(cfg.training.early_stopping_patience, cfg.training.early_stopping_min_delta, mode="max")
        self.scaler = torch.amp.GradScaler(enabled=(cfg.training.amp and self.device.type == "cuda"))
        self.wandb_logger = WandbLogger(cfg.wandb, cfg.name, config_to_dict(cfg))

        self.checkpoint_dir = Path(cfg.training.checkpoint_dir) / cfg.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_dir / "resolved_config.yaml", "w") as f:
            yaml.safe_dump(config_to_dict(cfg), f)

        self._backbone_frozen = cfg.training.frozen_epochs > 0
        self._set_backbone_frozen(self._backbone_frozen)

    def _validate_datasets(self) -> None:
        for source_name, source_cfg in self.cfg.data.sources_resolved.items():
            known = set(source_cfg.native_categories) if source_cfg.native_categories else None
            for split, path in source_cfg.annotations.items():
                report = validate_coco_file(path, known_category_names=known)
                report.raise_if_failed(f"{source_name}/{split}")
                for w in report.warnings:
                    print(f"[schema_validator] {source_name}/{split}: {w}")

    def _set_backbone_frozen(self, frozen: bool) -> None:
        for p in self.model.backbone_parameters():
            p.requires_grad = not frozen
        self._backbone_frozen = frozen

    def train(self) -> None:
        for epoch in range(self.cfg.training.max_epochs):
            if self._backbone_frozen and epoch >= self.cfg.training.frozen_epochs:
                self._set_backbone_frozen(False)
                print(f"[epoch {epoch}] backbone unfrozen")

            train_metrics = self._run_epoch(epoch, self.train_loader, train=True)
            log_images = (epoch % max(1, self.cfg.wandb.log_images_every_n_epochs)) == 0
            val_metrics = self._run_epoch(epoch, self.val_loader, train=False, log_images=log_images)

            log_dict: dict[str, Any] = {
                "train/loss": train_metrics["loss"],
                "train/macro_dice": train_metrics["pixel"]["macro"]["dice"],
                "train/macro_iou": train_metrics["pixel"]["macro"]["iou"],
                "val/loss": val_metrics["loss"],
                "val/macro_iou": val_metrics["pixel"]["macro"]["iou"],
                "val/macro_dice": val_metrics["pixel"]["macro"]["dice"],
                "val/freq_weighted_dice": val_metrics["pixel"]["frequency_weighted"]["dice"],
                "gap/macro_dice_train_minus_val": train_metrics["pixel"]["macro"]["dice"] - val_metrics["pixel"]["macro"]["dice"],
                "gap/loss_val_minus_train": val_metrics["loss"] - train_metrics["loss"],
                "lr_backbone": self.optimizer.param_groups[0]["lr"],
                "lr_head": self.optimizer.param_groups[1]["lr"],
                "backbone_frozen": int(self._backbone_frozen),
            }
            self.wandb_logger.log_epoch(epoch, log_dict)
            self.wandb_logger.log_per_class_table(epoch, "val", val_metrics["pixel"]["per_class"])
            print(f"[epoch {epoch}] ({train_metrics['seconds'] / 60:.1f} min train + {val_metrics['seconds'] / 60:.1f} min val) train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                  f"train_macro_dice={train_metrics['pixel']['macro']['dice']:.4f} val_macro_dice={val_metrics['pixel']['macro']['dice']:.4f}")

            monitor = log_dict["val/macro_dice"] if self.cfg.training.early_stopping_metric == "val_macro_dice" else -val_metrics["loss"]
            is_best = self.early_stopping.step(monitor)
            self._save_checkpoint(epoch, is_best)
            if self.early_stopping.should_stop:
                print(f"[epoch {epoch}] early stopping triggered (patience={self.cfg.training.early_stopping_patience})")
                break

        self.wandb_logger.finish()

    def _run_epoch(self, epoch: int, loader: DataLoader, train: bool, log_images: bool = False) -> dict[str, Any]:
        self.model.train(mode=train)
        acc = PixelMetricAccumulator(self.cfg.data.mvp_classes)
        total_loss, n_batches = 0.0, 0
        images_logged = False
        t_start = time.time()
        n_steps = len(loader)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for step, batch in enumerate(loader):
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                validity = batch["validity_mask"].to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device.type, enabled=(self.cfg.training.amp and self.device.type == "cuda")):
                    logits = self.model(images)
                    loss_dict = self.loss_fn(logits.float(), masks, validity)  # loss reductions in fp32 even under AMP
                    loss = loss_dict["total"]

                if train:
                    scaled = loss / self.cfg.training.grad_accum_steps
                    self.scaler.scale(scaled).backward()
                    if (step + 1) % self.cfg.training.grad_accum_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scheduler.step()

                probs = torch.sigmoid(logits.detach())
                acc.update(probs, masks, validity)
                total_loss += loss.item()
                n_batches += 1
                if train and (step + 1) % self.cfg.training.log_every_n_steps == 0:
                    rate = n_batches * loader.batch_size / (time.time() - t_start)
                    print(f"[epoch {epoch}] step {step + 1}/{n_steps} loss={total_loss / n_batches:.4f} ({rate:.0f} img/s)", flush=True)

                if (not train) and log_images and not images_logged:
                    k = min(self.cfg.wandb.num_logged_images, images.shape[0])
                    self.wandb_logger.log_images(epoch, images[:k], probs[:k], masks[:k], self.cfg.data.mvp_classes, IMAGENET_MEAN, IMAGENET_STD)
                    images_logged = True

        result = {"loss": total_loss / max(1, n_batches), "pixel": acc.compute(), "seconds": time.time() - t_start}
        return result

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": config_to_dict(self.cfg),
        }
        torch.save(state, self.checkpoint_dir / "last.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")
