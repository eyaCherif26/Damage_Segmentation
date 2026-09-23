"""A torchvision-backed MobileNetV3-Large encoder, registered for use by smp's decoders.

Why not smp's own MobileNetV3 (`tu-mobilenetv3_large_100`, via timm): the local dacl/bikit
checkpoints were trained on **torchvision**'s `mobilenet_v3_large`, whose parameter names share
nothing with timm's implementation — 0% of tensors matched, so `task_pretrained` was impossible
for this backbone. Using torchvision here makes the checkpoint's 308 encoder tensors map onto this
encoder exactly (verified: 308/308), while smp's FPN/Unet/DeepLabV3+ decoders keep working
because this class follows smp's encoder contract (`out_channels` + a forward returning one
feature per resolution, highest first).

Stage indices come from the torchvision layout (feature index -> stride):
1 -> /2 (16ch), 3 -> /4 (24ch), 6 -> /8 (40ch), 12 -> /16 (112ch), 16 -> /32 (960ch).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from segmentation_models_pytorch.encoders import encoders as SMP_ENCODER_REGISTRY
from segmentation_models_pytorch.encoders._base import EncoderMixin

ENCODER_NAME = "tv-mobilenet_v3_large"
OUT_CHANNELS = [3, 16, 24, 40, 112, 960]
OUT_INDEXES = [1, 3, 6, 12, 16]


class TorchvisionMobileNetV3LargeEncoder(nn.Module, EncoderMixin):
    def __init__(self, out_channels: list[int], depth: int = 5, output_stride: int = 32, imagenet_pretrained: bool = True) -> None:
        if not 1 <= depth <= 5:
            raise ValueError(f"depth must be in [1, 5], got {depth}")
        super().__init__()
        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride

        weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        self.features = torchvision.models.mobilenet_v3_large(weights=weights).features
        self._out_indexes = OUT_INDEXES

    def get_stages(self) -> dict[int, list[nn.Module]]:
        """Used by smp only for dilated (output_stride < 32) decoders."""
        return {16: [self.features[7:13]], 32: [self.features[13:]]}

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [x]
        for i, module in enumerate(self.features):
            x = module(x)
            if i in self._out_indexes:
                features.append(x)
            if len(features) > self._depth:
                break
        return features[: self._depth + 1]

    def load_state_dict(self, state_dict, **kwargs):  # type: ignore[override]
        for key in [k for k in state_dict if k.startswith("classifier.")]:
            state_dict.pop(key)
        return super().load_state_dict(state_dict, **kwargs)


def register() -> None:
    """Idempotent registration into smp's encoder registry.

    `pretrained_settings` stays empty and ImageNet weights are fetched by the encoder itself:
    smp's `get_encoder` resolves `encoder_weights` through a HuggingFace repo id, which this
    torchvision encoder has none of. Callers therefore pass `encoder_weights=None` and rely on
    `imagenet_pretrained` (see models/smp_model.py).
    """
    SMP_ENCODER_REGISTRY[ENCODER_NAME] = {
        "encoder": TorchvisionMobileNetV3LargeEncoder,
        "pretrained_settings": {},
        "params": {"out_channels": OUT_CHANNELS, "imagenet_pretrained": True},
    }


register()
