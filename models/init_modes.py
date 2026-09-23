"""Weight-init mode: orthogonal to backbone choice (see configs/schema.py::ModelConfig).

`from_scratch`: no-op — the registry factory already builds the model with ImageNet-pretrained
encoder weights (see models/smp_model.py, models/segformer_model.py). "from scratch" here means
"no *task*-pretraining", per PROMPT.md's own definition, not random-init.

`task_pretrained`: each model wrapper implements `load_task_pretrained()`, because the available
checkpoints differ per family — smp encoders take a bikit DaclNet *classifier* checkpoint
(models/bikit_adapter.py, encoder only), SegFormer takes a dacl10k-trained SegFormer
(models/segformer_pretrained.py, encoder + decode head). Both abort rather than partially load.
"""
from __future__ import annotations

import logging

import torch.nn as nn

from configs.schema import ModelConfig

logger = logging.getLogger(__name__)


def apply_init_mode(model: nn.Module, cfg: ModelConfig) -> dict | None:
    if cfg.init_mode == "from_scratch":
        return None
    elif cfg.init_mode == "task_pretrained":
        if not hasattr(model, "load_task_pretrained"):
            raise NotImplementedError(
                f"{type(model).__name__} does not implement load_task_pretrained(); add it to the "
                "wrapper (see models/smp_model.py or models/segformer_model.py)."
            )
        report = model.load_task_pretrained(
            cfg.task_pretrained_checkpoint, load_decode_head=cfg.load_decode_head, load_classifier=cfg.load_classifier
        )
        logger.info("Loaded task-pretrained weights: %s", report)
        return report
    else:
        raise ValueError(f"Unknown model.init_mode: {cfg.init_mode!r}")
