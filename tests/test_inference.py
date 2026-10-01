import numpy as np
import pytest
import torch
import torch.nn as nn

from inference.postprocess import apply_thresholds, export_coco_predictions
from inference.sliding_window import compute_tile_origins, run_tiled_inference
from inference.vram_budget import check_budget, estimate_accumulator_gb


class ConstantModel(nn.Module):
    def __init__(self, num_classes, value):
        super().__init__()
        self.num_classes, self.value = num_classes, value

    def forward(self, x):
        return torch.full((x.shape[0], self.num_classes, *x.shape[-2:]), self.value)


class CoordModel(nn.Module):
    """Returns logits that depend on absolute tile content (channel 0 = normalized red) so blending is checkable."""

    def forward(self, x):
        return x[:, :1].repeat(1, 2, 1, 1)


@pytest.mark.parametrize("w,h,tile,overlap", [(2200, 1300, 512, 64), (512, 512, 512, 64), (700, 300, 512, 128), (1025, 1025, 512, 0)])
def test_tiles_cover_every_pixel_and_stay_in_bounds(w, h, tile, overlap):
    cover = np.zeros((h, w), int)
    for o in compute_tile_origins(w, h, tile, overlap):
        assert o.x0 >= 0 and o.y0 >= 0
        cover[o.y0 : o.y0 + tile, o.x0 : o.x0 + tile] += 1
    assert cover.min() >= 1


def test_blending_of_constant_model_is_exactly_sigmoid_everywhere():
    img = np.full((700, 900, 3), 128, np.uint8)
    p = run_tiled_inference(ConstantModel(3, 1.5), img, 3, 256, 64, 4, torch.device("cpu"))
    assert p.shape == (3, 700, 900)
    assert np.allclose(p, 1 / (1 + np.exp(-1.5)), atol=1e-5), "overlap averaging must not double count"


def test_no_data_border_is_zeroed_and_valid_area_untouched():
    img = np.full((300, 300, 3), 200, np.uint8)
    img[:, :40] = 0
    p = run_tiled_inference(ConstantModel(2, 5.0), img, 2, 128, 32, 2, torch.device("cpu"))
    assert p[:, :, :40].max() == 0 and p[:, :, 40:].min() > 0.99


def test_tiled_matches_single_pass_for_pointwise_model():
    rng = np.random.default_rng(0)
    img = rng.integers(1, 255, (200, 330, 3), dtype=np.uint8)
    tiled = run_tiled_inference(CoordModel(), img, 2, 128, 48, 3, torch.device("cpu"))
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    norm = ((img / 255.0 - mean) / std)[..., 0]
    expected = 1 / (1 + np.exp(-norm))
    assert np.allclose(tiled[0], expected, atol=1e-4)


def test_budget_guard():
    assert estimate_accumulator_gb(5, 10_000, 10_000) > 2.0
    with pytest.raises(MemoryError):
        check_budget(5, 40_000, 40_000, max_memory_gb=8.0)
    check_budget(5, 40_000, 40_000, max_memory_gb=8.0, allow_large_memory=True)


def test_export_coco_filters_small_components_and_encodes_rle(tmp_path):
    classes = ["Cracks", "Rust"]
    prob = np.zeros((2, 100, 120), np.float32)
    prob[0, 10:30, 10:40] = 0.9
    prob[0, 80, 80] = 0.9  # 1 px speck
    binm = apply_thresholds(prob, classes, {"Cracks": 0.5})
    out = tmp_path / "p.json"
    export_coco_predictions(binm, prob, classes, 7, 120, 100, str(out), min_area=16)
    import json

    d = json.loads(out.read_text())
    assert len(d["annotations"]) == 1
    a = d["annotations"][0]
    assert a["category_id"] == 1 and a["image_id"] == 7 and a["area"] == 600
    assert a["bbox"] == [10.0, 10.0, 30.0, 20.0] and abs(a["score"] - 0.9) < 1e-5
    from pycocotools import mask as mu

    assert mu.decode({"size": a["segmentation"]["size"], "counts": a["segmentation"]["counts"].encode()}).sum() == 600
