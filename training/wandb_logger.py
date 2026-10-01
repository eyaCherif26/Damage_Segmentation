"""W&B experiment tracking — replaces the MLflow pattern used everywhere else in sibling
projects (deliberate, standalone departure; see CLAUDE.md, no dual-logging).

Logs per-epoch train AND val curves (not just final numbers) plus per-class metric tables every
epoch specifically so a growing train/val gap — the symptom this project is designed to catch
early — is visible during the run, not discovered post-hoc.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from configs.schema import WandbConfig


class WandbLogger:
    def __init__(self, cfg: WandbConfig, run_name: str, config_dict: dict[str, Any]) -> None:
        self.cfg = cfg
        self.enabled = cfg.mode != "disabled"
        self.run = None
        if self.enabled:
            cfg.validate_ready()
            import wandb

            self._wandb = wandb
            self.run = wandb.init(
                entity=cfg.entity,
                project=cfg.project,
                name=cfg.run_name or run_name,
                config=config_dict,
                mode=cfg.mode,
            )

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        if self.enabled:
            self._wandb.log(metrics, step=epoch)

    def log_per_class_table(self, epoch: int, split: str, per_class: dict[str, dict[str, float]]) -> None:
        if not self.enabled:
            return
        table = self._wandb.Table(columns=["class", "iou", "dice", "precision", "recall"])
        for name, m in per_class.items():
            table.add_data(name, m["iou"], m["dice"], m["precision"], m["recall"])
        self._wandb.log({f"{split}/per_class_table": table}, step=epoch)

    def log_images(
        self,
        epoch: int,
        images: torch.Tensor,
        pred_probs: torch.Tensor,
        target: torch.Tensor,
        class_names: list[str],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        if not self.enabled:
            return
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        logged = []
        for i in range(images.shape[0]):
            img = (images[i].cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
            img_uint8 = (img * 255).astype(np.uint8)
            pred_mask = (pred_probs[i].cpu().numpy() >= 0.5).astype(np.uint8)
            gt_mask = target[i].cpu().numpy().astype(np.uint8)
            masks = {
                "prediction": {"mask_data": _flatten_channels(pred_mask), "class_labels": {j + 1: n for j, n in enumerate(class_names)}},
                "ground_truth": {"mask_data": _flatten_channels(gt_mask), "class_labels": {j + 1: n for j, n in enumerate(class_names)}},
            }
            logged.append(self._wandb.Image(img_uint8, masks=masks))
        self._wandb.log({"val/predictions": logged}, step=epoch)

    def log_artifact(
        self,
        path: str,
        name: str,
        type_: str,
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        if not self.enabled:
            return
        artifact = self._wandb.Artifact(name, type=type_, metadata=metadata or {})
        artifact.add_file(path)
        self._wandb.log_artifact(artifact, aliases=aliases)

    def finish(self) -> None:
        if self.enabled:
            self._wandb.finish()


def _flatten_channels(multi_channel_mask: np.ndarray) -> np.ndarray:
    """(C, H, W) multi-label binary -> (H, W) single-label for W&B's mask overlay (which expects
    one label per pixel); a pixel with multiple active classes gets the highest channel index —
    display-only simplification, not used for any actual metric computation."""
    h, w = multi_channel_mask.shape[1:]
    flat = np.zeros((h, w), dtype=np.uint8)
    for c in range(multi_channel_mask.shape[0]):
        flat = np.where(multi_channel_mask[c] > 0, c + 1, flat)
    return flat
