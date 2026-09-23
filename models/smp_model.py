"""segmentation_models_pytorch-backed models. The decoder is config-selectable (`model.decoder`: fpn | unet |
deeplabv3plus); default FPN because FPN + EfficientNet was the best CNN baseline in the dacl10k paper. Two registry
entries for the MVP:

- EfficientNet-B0 uses smp's *native* `efficientnet-b0` encoder (wraps the `efficientnet_pytorch`
  package), deliberately NOT the `tu-efficientnet_b0` (timm) variant, because bikit's own DaclNet
  checkpoints build EfficientNet-B0 via the same `efficientnet_pytorch` package — verified
  directly (99.4% state-dict tensor coverage, 358/360 tensors, against
  `bikit_models/.../mcds_bikit_EfficientNetV1-B0_dha.pth`) — making `task_pretrained` mode
  actually load real dacl-pretrained weights for this backbone, not just pass the config guard.
- MobileNetV3-Large uses the `tu-` (timm-universal) prefix (matches the dacl10k paper's actual
  backbone choice, and lets `drop_rate` pass through for dropout). `task_pretrained` is NOT
  currently functional for this backbone: the only local MobileNet-family bikit checkpoints are
  either MobileNetV3-Large via torchvision (0% key-structure match against timm's
  implementation) or a single non-dacl MobileNetV2 checkpoint (0% match due to a state-dict
  nesting-depth mismatch against smp's native mobilenet_v2 encoder, despite both wrapping the
  same torchvision architecture) — verified directly, both correctly rejected by the coverage
  guard rather than silently loading garbage. See CLAUDE.md.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from configs.schema import ModelConfig
from models.bikit_adapter import load_bikit_encoder_into
from models.registry import register_model
from models.torchvision_encoders import ENCODER_NAME as TV_MOBILENET_ENCODER

# encoder_name -> underlying smp encoder id
_ENCODER_IDS = {
    "efficientnet-b0": "efficientnet-b0",  # native (efficientnet_pytorch) — matches bikit checkpoints
    "mobilenet": TV_MOBILENET_ENCODER,  # torchvision MobileNetV3-Large — matches bikit checkpoints
}

# bikit wraps torchvision models as nn.Sequential(features, avgpool), losing the "features" name.
_BIKIT_KEY_REWRITES = {TV_MOBILENET_ENCODER: ((r"^0\.", "features."),)}


_DECODERS = {"fpn": smp.FPN, "unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}


class SmpSegModel(nn.Module):
    """Wraps an smp segmentation model, returning raw logits at input resolution (sigmoid applied
    downstream in the loss/metric layer, matching the SegFormer wrapper's convention)."""

    def __init__(self, encoder_id: str, decoder: str, num_classes: int, dropout: float, pretrained: bool) -> None:
        super().__init__()
        self.encoder_id = encoder_id
        # The torchvision encoder fetches its own ImageNet weights (smp resolves `encoder_weights`
        # through a HuggingFace repo id, which it does not have).
        encoder_weights = None if encoder_id == TV_MOBILENET_ENCODER else ("imagenet" if pretrained else None)
        self.net = _DECODERS[decoder](
            encoder_name=encoder_id,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None,
        )
        # Dropout goes in front of the segmentation head via a forward pre-hook rather than by wrapping
        # the head in a new Sequential: wrapping shifts every parameter key (`segmentation_head.0.*` ->
        # `.1.*`) and silently makes all earlier checkpoints unloadable. Dropout2d has no parameters, so
        # the state_dict layout is identical for any `dropout` value (pinned in tests).
        self.head_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.net.segmentation_head.register_forward_pre_hook(lambda _module, args: (self.head_dropout(args[0]),))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values)

    def load_task_pretrained(self, checkpoint_path: str, load_decode_head: bool = True, load_classifier: bool = False) -> dict:
        # bikit checkpoints are classifiers: only the encoder transfers, there is no decoder to load.
        return load_bikit_encoder_into(self, checkpoint_path, key_rewrites=_BIKIT_KEY_REWRITES.get(self.encoder_id, ()))

    def backbone_parameters(self):
        return self.net.encoder.parameters()

    def head_parameters(self):
        for module in (self.net.decoder, self.net.segmentation_head):
            yield from module.parameters()


def _build(model_cfg: ModelConfig, encoder_key: str) -> SmpSegModel:
    encoder_id = _ENCODER_IDS[encoder_key]
    # pretrained=True always at construction time (from_scratch == "ImageNet-pretrained encoder
    # only", per PROMPT.md's definition — "from scratch" means no *task*-pretraining, not random
    # init); task_pretrained overwrites these weights afterward via models/init_modes.py.
    return SmpSegModel(encoder_id, model_cfg.decoder, model_cfg.num_classes, model_cfg.dropout, pretrained=True)


@register_model("smp_efficientnet_b0")
def build_smp_efficientnet_b0(cfg: ModelConfig) -> SmpSegModel:
    return _build(cfg, "efficientnet-b0")


@register_model("smp_mobilenet")
def build_smp_mobilenet(cfg: ModelConfig) -> SmpSegModel:
    return _build(cfg, "mobilenet")
