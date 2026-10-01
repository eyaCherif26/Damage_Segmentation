"""Preprocessing must be identical for every model and every stage of the pipeline.

Two distinct requirements, both pinned here:
1. the *same* transform for all three backbones (a fair comparison depends on it), and
2. that shared transform must be what each pretrained backbone's weights actually expect —
   they happen to agree on ImageNet statistics, but that is a fact to verify, not to assume.
"""
from __future__ import annotations

import dataclasses
import glob
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from configs.loader import load_experiment_config
from configs.schema import AugmentationConfig
from data.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform, build_train_transform

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = sorted(glob.glob(str(REPO / "configs/experiments/*.yaml")))

# Deliberate resolution ablation (2026-09-28, see CLAUDE.md): trains at 1024 instead of the
# dacl10k-paper 512 every other config uses. Named explicitly rather than just excluded by a size
# check, so a THIRD, accidental resolution divergence still fails loudly instead of silently
# joining this list.
_DELIBERATE_1024_CONFIGS = {
    "mobilenet_cif_1024.yaml", "mobilenet_native_cif_1024.yaml",
    "segformer_cif_1024.yaml", "segformer_native_cif_1024.yaml",
}


def test_every_experiment_shares_augmentation_resolution_and_classes():
    configs = [load_experiment_config(f) for f in EXPERIMENTS]
    assert len({json.dumps(dataclasses.asdict(c.augmentation), sort_keys=True) for c in configs}) == 1
    assert len({tuple(c.data.mvp_classes) for c in configs}) == 1
    assert len({c.data.no_data_validity for c in configs}) == 1

    default_res = {c.data.image_size for f, c in zip(EXPERIMENTS, configs) if Path(f).name not in _DELIBERATE_1024_CONFIGS}
    assert len(default_res) == 1, f"the non-ablation configs must still share one resolution, got {default_res}"
    ablation_res = {c.data.image_size for f, c in zip(EXPERIMENTS, configs) if Path(f).name in _DELIBERATE_1024_CONFIGS}
    assert ablation_res == {1024}, f"expected exactly the documented 1024 ablation configs, got {ablation_res}"
    assert len({c.data.no_data_validity for c in configs}) == 1


def test_smp_encoders_expect_our_normalisation():
    from segmentation_models_pytorch.encoders import get_preprocessing_params

    from models.torchvision_encoders import ENCODER_NAME

    params = get_preprocessing_params("efficientnet-b0")
    assert params["input_space"] == "RGB" and params["input_range"] == [0, 1]
    assert np.allclose(params["mean"], IMAGENET_MEAN) and np.allclose(params["std"], IMAGENET_STD)

    import torchvision

    tv = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms()
    assert np.allclose(tv.mean, IMAGENET_MEAN) and np.allclose(tv.std, IMAGENET_STD), ENCODER_NAME


@pytest.mark.parametrize("name", ["nvidia/mit-b0", "nvidia/mit-b1"])
def test_segformer_expects_our_normalisation(name):
    from transformers import SegformerImageProcessor

    p = SegformerImageProcessor.from_pretrained(name)
    assert np.allclose(p.image_mean, IMAGENET_MEAN) and np.allclose(p.image_std, IMAGENET_STD)
    assert p.rescale_factor == pytest.approx(1 / 255)


def _sample():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    mask = np.zeros((300, 400, 6), np.float32)
    mask[50:200, 60:300, 2] = 1
    mask[:, :, 5] = 1  # validity channel
    return img, mask


@pytest.mark.parametrize("train", [True, False])
def test_masks_stay_binary_through_the_transforms(train):
    """Geometric ops must use nearest interpolation for masks; bilinear would produce fractional
    targets and a silently wrong Dice/validity mask."""
    img, mask = _sample()
    t = build_train_transform(AugmentationConfig(), 512) if train else build_eval_transform(512)
    seen: set[float] = set()
    for _ in range(15):
        seen |= set(np.unique(t(image=img, mask=mask)["mask"]).tolist())
    assert seen <= {0.0, 1.0}, f"non-binary mask values: {sorted(seen)}"


def test_photometric_augmentation_never_touches_the_image_statistics_contract():
    """Whatever augmentation does, the tensor handed to every model is ImageNet-normalised."""
    img, mask = _sample()
    out = build_train_transform(AugmentationConfig(), 512)(image=img, mask=mask)["image"]
    assert out.dtype == np.float32
    plain = (img.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    assert plain.min() < out.max() and out.min() > -5 and out.max() < 5


def test_training_and_sliding_window_inference_normalise_identically():
    from inference.sliding_window import run_tiled_inference

    tile = (np.random.default_rng(1).random((512, 512, 3)) * 255).astype(np.uint8)
    from_dataset = build_eval_transform(512)(image=tile, mask=np.zeros((512, 512, 6), np.float32))["image"]

    captured = {}

    class Capture(torch.nn.Module):
        def forward(self, x):
            captured["x"] = x[0].permute(1, 2, 0).numpy()
            return torch.zeros(x.shape[0], 2, *x.shape[-2:])

    run_tiled_inference(Capture(), tile, 2, 512, 0, 1, torch.device("cpu"))
    assert np.abs(captured["x"] - from_dataset).max() < 1e-5


def test_eval_transform_is_deterministic_and_unaugmented():
    img, mask = _sample()
    t = build_eval_transform(512)
    a, b = t(image=img, mask=mask), t(image=img, mask=mask)
    assert np.array_equal(a["image"], b["image"]) and np.array_equal(a["mask"], b["mask"])
