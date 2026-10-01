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
from training.amp_config import resolve_amp
from training.early_stopping import EarlyStopping
from training.param_groups import build_param_groups
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
        # 4 groups (backbone/head x decay/no-decay), not 2: biases and norm-layer scale/offset
        # params are excluded from weight decay (standard AdamW practice; see training/param_groups.py).
        # weight_decay is passed per-group below, not to AdamW's own `weight_decay=` kwarg.
        self.optimizer = torch.optim.AdamW(
            build_param_groups(self.model, backbone_params, head_params, cfg.training.lr_backbone,
                               cfg.training.lr_head, cfg.training.weight_decay),
        )
        self._all_trainable_params = [p for group in self.optimizer.param_groups for p in group["params"]]
        steps_per_epoch = max(1, len(self.train_loader) // cfg.training.grad_accum_steps)
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer, cfg.training.warmup_epochs, cfg.training.max_epochs, steps_per_epoch, cfg.training.min_lr
        )

        self.early_stopping = EarlyStopping(cfg.training.early_stopping_patience, cfg.training.early_stopping_min_delta, mode="max")

        bf16_supported = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        amp_settings = resolve_amp(cfg.training.amp, cfg.training.amp_dtype, self.device.type, bf16_supported)
        self._amp_enabled, self._autocast_dtype = amp_settings.enabled, amp_settings.autocast_dtype
        if amp_settings.warning:
            print(f"[trainer] WARNING: {amp_settings.warning}", flush=True)
        self.scaler = torch.amp.GradScaler(enabled=amp_settings.scaler_enabled)
        self.wandb_logger = WandbLogger(cfg.wandb, cfg.name, config_to_dict(cfg))

        self.checkpoint_dir = Path(cfg.training.checkpoint_dir) / cfg.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_dir / "resolved_config.yaml", "w") as f:
            yaml.safe_dump(config_to_dict(cfg), f)

        self._backbone_frozen = cfg.training.frozen_epochs > 0
        self._set_backbone_frozen(self._backbone_frozen)
        self._consecutive_nonfinite_batches = 0  # persists across epochs; see _run_epoch's finite-loss guard
        self._best_epoch: int | None = None
        self._best_monitor_value: float | None = None

    def _validate_datasets(self) -> None:
        for source_name, source_cfg in self.cfg.data.sources_resolved.items():
            known = set(source_cfg.native_categories) if source_cfg.native_categories else None
            for split, path in source_cfg.annotations.items():
                report = validate_coco_file(path, known_category_names=known)
                report.raise_if_failed(f"{source_name}/{split}")
                for w in report.warnings:
                    print(f"[schema_validator] {source_name}/{split}: {w}")

    def _lr_by_group_name(self, prefix: str) -> float:
        """LR of the "backbone" or "head" param group. Decay/no-decay share one LR per prefix
        (build_param_groups), so either sub-group's value is correct; falls back to the
        "_no_decay" sibling on the rare model where the plain group ended up empty and omitted."""
        for group in self.optimizer.param_groups:
            if group["name"] == prefix or group["name"] == f"{prefix}_no_decay":
                return group["lr"]
        raise KeyError(f"No optimizer param group named {prefix!r} or {prefix!r}_no_decay")

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
                # Find each group by its "name" rather than by index: build_param_groups can produce
                # up to 4 groups (backbone/head x decay/no-decay), and a group is omitted entirely
                # when empty, so neither "backbone" nor "head" is guaranteed to sit at a fixed index.
                "lr_backbone": self._lr_by_group_name("backbone"),
                "lr_head": self._lr_by_group_name("head"),
                "backbone_frozen": int(self._backbone_frozen),
            }
            self.wandb_logger.log_epoch(epoch, log_dict)
            self.wandb_logger.log_per_class_table(epoch, "val", val_metrics["pixel"]["per_class"])
            print(f"[epoch {epoch}] ({train_metrics['seconds'] / 60:.1f} min train + {val_metrics['seconds'] / 60:.1f} min val) train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                  f"train_macro_dice={train_metrics['pixel']['macro']['dice']:.4f} val_macro_dice={val_metrics['pixel']['macro']['dice']:.4f}")

            monitor = log_dict["val/macro_dice"] if self.cfg.training.early_stopping_metric == "val_macro_dice" else -val_metrics["loss"]
            is_best = self.early_stopping.step(monitor)
            self._save_checkpoint(epoch, is_best)
            if is_best:
                self._best_epoch, self._best_monitor_value = epoch, monitor
            if self.early_stopping.should_stop:
                print(f"[epoch {epoch}] early stopping triggered (patience={self.cfg.training.early_stopping_patience})")
                break

        self._log_best_checkpoint_artifact()
        self.wandb_logger.finish()

    def _log_best_checkpoint_artifact(self) -> None:
        """Uploads the run's best checkpoint to W&B once, at the end of training (not every time a
        new best is found mid-run — that would re-upload a multi-hundred-MB file on every
        improvement, which is wasteful for how often val Dice improves early in a run). Tagged with
        the best epoch/metric and a stable 'best' alias so it's fetchable without knowing the
        version number. No-op if W&B logging is disabled (WandbLogger.log_artifact already checks
        this) or if no checkpoint was ever saved (e.g. max_epochs=0)."""
        best_path = self.checkpoint_dir / "best.pt"
        if self._best_epoch is None or not best_path.exists():
            return
        self.wandb_logger.log_artifact(
            str(best_path),
            name=f"{self.cfg.name}-best",
            type_="model",
            metadata={"epoch": self._best_epoch, self.cfg.training.early_stopping_metric: self._best_monitor_value},
            aliases=["best"],
        )

    def _run_epoch(self, epoch: int, loader: DataLoader, train: bool, log_images: bool = False) -> dict[str, Any]:
        self.model.train(mode=train)
        acc = PixelMetricAccumulator(self.cfg.data.mvp_classes)
        total_loss, n_batches, n_skipped = 0.0, 0, 0
        images_logged = False
        t_start = time.time()
        n_steps = len(loader)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for step, batch in enumerate(loader):
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                validity = batch["validity_mask"].to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype, enabled=self._amp_enabled):
                    logits = self.model(images)
                    loss_dict = self.loss_fn(logits.float(), masks, validity)  # loss reductions in fp32 even under AMP
                    loss = loss_dict["total"]

                # A non-finite loss (nan/inf) carries no usable gradient — most often fp16 autocast
                # overflow (max ~65504) somewhere in the model's own forward pass, not the loss math
                # (already fp32-guarded above). Skip the batch entirely: no backward/optimizer step,
                # not counted into the epoch's loss/metric averages (previously a single such batch
                # silently poisoned the rest of the epoch's printed running-average loss). This can
                # only limit the damage of THIS batch, not undo it: if the forward pass already wrote
                # a non-finite value into a BatchNorm running-stat buffer, that happened before this
                # check runs and is not reversible here (see CLAUDE.md's Pitfalls) — hence the
                # consecutive-count abort below, so a genuinely corrupted model fails loudly instead
                # of burning epochs producing nan/0.0 metrics until early stopping eventually notices.
                if train and not torch.isfinite(loss):
                    self._consecutive_nonfinite_batches += 1
                    print(f"[epoch {epoch}] step {step + 1}/{n_steps}: non-finite loss ({loss.item()!r}) — "
                          f"skipping this batch, no weight update (image_ids={batch.get('image_id')})", flush=True)
                    if self._consecutive_nonfinite_batches >= self.cfg.training.max_consecutive_nonfinite_batches:
                        raise RuntimeError(
                            f"{self._consecutive_nonfinite_batches} consecutive non-finite-loss batches "
                            f"(training.max_consecutive_nonfinite_batches={self.cfg.training.max_consecutive_nonfinite_batches}) "
                            "— the model is very likely permanently corrupted (e.g. a BatchNorm running-stat "
                            "buffer poisoned by an earlier fp16 overflow), not hitting isolated bad batches. "
                            "Resume from the last good checkpoint; consider bf16 autocast to prevent the overflow."
                        )
                    n_skipped += 1
                    continue
                self._consecutive_nonfinite_batches = 0

                if train:
                    scaled = loss / self.cfg.training.grad_accum_steps
                    self.scaler.scale(scaled).backward()
                    if (step + 1) % self.cfg.training.grad_accum_steps == 0:
                        if self.cfg.training.grad_clip_max_norm is not None:
                            # Must unscale before clipping under AMP, or the norm is computed on the
                            # scaled (much larger) gradients and the clip threshold means nothing.
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self._all_trainable_params, self.cfg.training.grad_clip_max_norm)
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

        if n_skipped:
            print(f"[epoch {epoch}] skipped {n_skipped}/{n_steps} batches with non-finite loss", flush=True)
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
