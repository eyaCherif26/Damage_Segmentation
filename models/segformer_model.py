"""HuggingFace SegFormer wrapper.

Default backbone: MiT-b0 (3.7M-param encoder) — smaller than the Dacl-Challenge baseline's
MiT-b1 (13.1M), a deliberate `[Design choice]` departure from the literature setting to keep
headroom on the 8GB VRAM budget while three architectures are compared; `model.encoder_name` is
config-selectable (e.g. `nvidia/mit-b1`) if closer literature match is wanted for a given run.

`SegformerForSemanticSegmentation`'s decode head outputs logits at H/4 x W/4 — this wrapper
upsamples back to input resolution so every registered model satisfies the same
full-resolution-logits contract (see models/registry.py).

`init_mode: task_pretrained` loads a dacl10k-trained MiT checkpoint (encoder + decode head, fresh
classifier) via models/segformer_pretrained.py — the local one is MiT-b1, so pair it with
`encoder_name: nvidia/mit-b1`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from configs.schema import ModelConfig
from models.registry import register_model
from models.segformer_pretrained import load_segformer_pretrained

DEFAULT_ENCODER = "nvidia/mit-b0"


class SegFormerSegModel(nn.Module):
    def __init__(self, encoder_name: str, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            encoder_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout / 2,  # attention dropout is typically kept lower than hidden dropout
            classifier_dropout_prob=dropout,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        low_res_logits = self.net(pixel_values=pixel_values).logits
        return F.interpolate(low_res_logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)

    def load_task_pretrained(self, checkpoint_path: str, load_decode_head: bool = True, load_classifier: bool = False) -> dict:
        return load_segformer_pretrained(self.net, checkpoint_path, load_decode_head=load_decode_head, load_classifier=load_classifier)

    def backbone_parameters(self):
        # `.segformer` is the whole MiT encoder (transformers 5.x nests it under `.stages`; 4.x under
        # `.encoder`), so take the parent module — its parameters plus `.decode_head`'s are exhaustive.
        return self.net.segformer.parameters()

    def head_parameters(self):
        return self.net.decode_head.parameters()


@register_model("segformer")
def build_segformer(cfg: ModelConfig) -> SegFormerSegModel:
    encoder_name = cfg.encoder_name or DEFAULT_ENCODER
    return SegFormerSegModel(encoder_name, cfg.num_classes, cfg.dropout)
