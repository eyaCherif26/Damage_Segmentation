import json

import cv2
import numpy as np
import pytest
import torch
import yaml
from pycocotools import mask as mask_utils

from configs.schema import DataConfig
from data.datasets.combined_dataset import _make_dataset
from data.datasets.crop_dataset import CocoCropDataset
from data.datasets.multilabel_seg_dataset import CocoMultiLabelSegDataset

MVP = ["Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence"]
W, H = 400, 300
# a bright rectangle in the image, and the label polygon that outlines exactly it
RECT = (150, 100, 250, 200)  # x0, y0, x1, y1
POLY = [[RECT[0], RECT[1], RECT[2], RECT[1], RECT[2], RECT[3], RECT[0], RECT[3]]]


def _image(w=W, h=H):
    rng = np.random.default_rng(0)
    img = rng.integers(40, 100, (h, w, 3), dtype=np.uint8)  # dim textured background
    if (w, h) == (W, H):
        img[RECT[1] : RECT[3], RECT[0] : RECT[2]] = (250, 240, 30)  # bright rectangle
    return img


def _write(tmp_path, img=None, anns=None, coco_wh=None, name="a.jpg"):
    img = _image() if img is None else img
    cv2.imwrite(str(tmp_path / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
    ch, cw = (img.shape[0], img.shape[1]) if coco_wh is None else (coco_wh[1], coco_wh[0])
    if anns is None:
        anns = [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": POLY, "area": 100 * 100,
                 "bbox": [RECT[0], RECT[1], 100, 100], "iscrowd": 0}]
    coco = {"images": [{"id": 1, "file_name": name, "width": cw, "height": ch}],
            "categories": [{"id": 1, "name": "Rust"}, {"id": 2, "name": "Spalling"}], "annotations": anns}
    (tmp_path / "coco.json").write_text(json.dumps(coco))
    (tmp_path / "map.yaml").write_text(yaml.safe_dump({"Rust": "Rust", "Spalling": "Spalling"}))
    return tmp_path


def _ds(root, **kw):
    kw.setdefault("crop_size", 64)
    kw.setdefault("mode", "fixed")
    return CocoCropDataset(root, root / "coco.json", root / "map.yaml", MVP, **kw)


# ------------------------------------------------------------------------------------- geometry / alignment
def test_scale_one_crop_is_an_exact_pixel_window_of_the_native_image(tmp_path):
    root = _write(tmp_path)
    ds = _ds(root, scale=1.0)
    img, masks, pad = ds.render(1, ox=120.0, oy=80.0, scale=1.0)
    native = _image()
    # JPEG round-trip: compare against the decoded file, not the in-memory array
    decoded = cv2.cvtColor(cv2.imread(str(root / "a.jpg")), cv2.COLOR_BGR2RGB)
    assert np.array_equal(img, decoded[80:144, 120:184]), "scale 1.0 must be a bit-exact window"
    assert pad.min() == 1.0 and native.shape[:2] == (H, W)


def test_labels_match_a_direct_native_rasterisation_cropped(tmp_path):
    root = _write(tmp_path)
    ds = _ds(root, scale=1.0)
    _, masks, _ = ds.render(1, ox=120.0, oy=80.0, scale=1.0)
    ref = CocoMultiLabelSegDataset(root, root / "coco.json", root / "map.yaml", MVP)[0]["mask"].numpy()  # native (no transform)
    assert (masks[:, :, 3] != ref[3][80:144, 120:184]).sum() <= 12, "crop labels == native labels cropped (edge rounding only)"
    assert masks[:, :, [0, 1, 2, 4]].sum() == 0 and masks[:, :, 3].sum() > 0


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.3, 1.5])
def test_labels_stay_glued_to_the_image_content_at_any_scale(tmp_path, scale):
    """The bright rectangle is in the image AND outlined by the label. After scaling+cropping the two must
    still coincide — the failure mode this dataset must never have."""
    root = _write(tmp_path)
    ds = _ds(root, crop_size=96, scale=scale)
    ox, oy = max(0.0, 200 * scale - 48), max(0.0, 150 * scale - 48)  # window centred on the rectangle
    img, masks, pad = ds.render(1, ox, oy, scale)
    bright = (img[:, :, 0] > 200) & (img[:, :, 2] < 120)
    label = masks[:, :, 3] > 0.5
    assert bright.sum() > 100 and label.sum() > 100
    iou = (bright & label).sum() / (bright | label).sum()
    assert iou > 0.85, f"scale {scale}: image content and label disagree (IoU {iou:.2f})"


def test_rle_annotations_agree_with_the_polygon_path(tmp_path):
    poly_mask = np.zeros((H, W), np.uint8)
    poly_mask[RECT[1] : RECT[3], RECT[0] : RECT[2]] = 1
    rle = mask_utils.encode(np.asfortranarray(poly_mask))
    rle["counts"] = rle["counts"].decode()
    rle_ann = [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": rle, "area": 10000, "bbox": [150, 100, 100, 100], "iscrowd": 0}]
    (tmp_path / "rle").mkdir()
    (tmp_path / "poly").mkdir()
    ds_rle = _ds(_write(tmp_path / "rle", anns=rle_ann), crop_size=96, scale=0.5)
    ds_poly = _ds(_write(tmp_path / "poly"), crop_size=96, scale=0.5)
    m_rle = ds_rle.render(1, 52.0, 27.0, 0.5)[1][:, :, 3] > 0
    m_poly = ds_poly.render(1, 52.0, 27.0, 0.5)[1][:, :, 3] > 0
    assert m_rle.sum() > 500 and (m_rle & m_poly).sum() / (m_rle | m_poly).sum() > 0.9


def test_annotations_outside_the_window_are_ignored_and_multilabel_overlap_survives(tmp_path):
    anns = [
        {"id": 1, "image_id": 1, "category_id": 1, "segmentation": POLY, "area": 1, "bbox": [150, 100, 100, 100], "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 2, "segmentation": [[160, 110, 200, 110, 200, 150, 160, 150]], "area": 1, "bbox": [160, 110, 40, 40], "iscrowd": 0},
        {"id": 3, "image_id": 1, "category_id": 2, "segmentation": [[5, 5, 30, 5, 30, 30, 5, 30]], "area": 1, "bbox": [5, 5, 25, 25], "iscrowd": 0},
    ]
    root = _write(tmp_path, anns=anns)
    ds = _ds(root, crop_size=96, scale=1.0)
    _, masks, _ = ds.render(1, 130.0, 90.0, 1.0)
    assert ((masks[:, :, 3] > 0) & (masks[:, :, 2] > 0)).sum() > 500, "Rust-inside-Spalling stays in both channels"
    _, far, _ = ds.render(1, 300.0, 200.0, 1.0)
    assert far.sum() == 0, "a window containing no annotation has empty masks"


# ------------------------------------------------------------------------------------- padding / no-data
def test_window_beyond_the_image_is_padded_and_marked_invalid(tmp_path):
    small = _image(50, 40)
    root = _write(tmp_path, img=small, anns=[])
    ds = _ds(root, crop_size=64, scale=1.0)
    item = ds[0]
    v = item["validity_mask"][0].numpy()
    assert v[:40, :50].min() == 1 and v[40:, :].max() == 0 and v[:, 50:].max() == 0, "image area valid, padding invalid"
    assert item["image"].shape == (3, 64, 64)


def test_black_border_inside_a_crop_is_flagged_as_no_data(tmp_path):
    img = _image()
    img[:, :100] = 0  # black padding on the left, as in cif_coco
    root = _write(tmp_path, img=img, anns=[])
    ds = _ds(root, crop_size=200, scale=1.0, mode="grid", detect_no_data=True)  # tile 0 spans x = 0..199
    v = ds[0]["validity_mask"][0].numpy()
    assert v[:, :90].max() == 0, "the black border is excluded"
    assert v[:, 110:].min() == 1, "real image content stays valid"
    off = _ds(root, crop_size=200, scale=1.0, mode="grid", detect_no_data=False)[0]["validity_mask"][0].numpy()
    assert off.min() == 1, "detection is opt-in"


# ------------------------------------------------------------------------------------- grid mode
@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_grid_tiles_cover_every_pixel_exactly_once(tmp_path, scale):
    root = _write(tmp_path)
    ds = _ds(root, crop_size=96, scale=scale, mode="grid")
    ws, hs = round(W * scale), round(H * scale)
    assert len(ds) == int(np.ceil(ws / 96)) * int(np.ceil(hs / 96))
    valid_px = sum(int(ds[i]["validity_mask"].sum()) for i in range(len(ds)))
    assert abs(valid_px - ws * hs) <= 0.02 * ws * hs, "tiles partition the scaled image"
    if scale == 1.0:
        assert valid_px == W * H
        total_label = sum(float(ds[i]["mask"][3].sum()) for i in range(len(ds)))
        assert abs(total_label - 100 * 100) <= 60, "every labelled pixel appears in exactly one tile"


def test_grid_reconstructs_the_native_image_at_scale_one(tmp_path):
    root = _write(tmp_path)
    ds = _ds(root, crop_size=96, scale=1.0, mode="grid")
    decoded = cv2.cvtColor(cv2.imread(str(root / "a.jpg")), cv2.COLOR_BGR2RGB)
    canvas = np.zeros_like(decoded)
    for i in range(len(ds)):
        _, tx, ty = ds._tiles[i]
        tile, _, pad = ds.render(1, tx * 96.0, ty * 96.0, 1.0)
        h, w = min(96, H - ty * 96), min(96, W - tx * 96)
        canvas[ty * 96 : ty * 96 + h, tx * 96 : tx * 96 + w] = tile[:h, :w]
    assert np.array_equal(canvas, decoded)


def test_sample_image_id_maps_tiles_and_crops_back_to_images(tmp_path):
    root = _write(tmp_path)
    grid = _ds(root, crop_size=96, mode="grid")
    assert {grid.sample_image_id(i) for i in range(len(grid))} == {1}
    many = _ds(root, crops_per_image=5)
    assert len(many) == 5 and {many.sample_image_id(i) for i in range(5)} == {1}


# ------------------------------------------------------------------------------------- sampling behaviour
def test_fixed_mode_is_deterministic_and_seed_dependent(tmp_path):
    root = _write(tmp_path)
    a, b = _ds(root, crop_size=64, mode="fixed", fg_crop_prob=0.0, seed=1), _ds(root, crop_size=64, mode="fixed", fg_crop_prob=0.0, seed=1)
    assert torch.equal(a[0]["image"], b[0]["image"]) and torch.equal(a[0]["image"], a[0]["image"])
    c = _ds(root, crop_size=64, mode="fixed", fg_crop_prob=0.0, seed=2)
    assert not torch.equal(a[0]["image"], c[0]["image"])


def test_random_mode_varies_and_is_reproducible_from_the_torch_seed(tmp_path):
    root = _write(tmp_path)

    def draw():
        torch.manual_seed(123)
        ds = _ds(root, crop_size=64, mode="random", fg_crop_prob=0.0)
        return [ds[0]["image"] for _ in range(6)]

    first, second = draw(), draw()
    assert all(torch.equal(x, y) for x, y in zip(first, second)), "same seed -> same crops"
    assert len({x.sum().item() for x in first}) > 3, "successive crops differ"


def test_foreground_bias_centres_crops_on_objects(tmp_path):
    big = _image(1600, 1200)
    anns = [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[800, 600, 840, 600, 840, 640, 800, 640]], "area": 1600,
             "bbox": [800, 600, 40, 40], "iscrowd": 0}]
    root = _write(tmp_path, img=big, anns=anns)

    def hit_rate(p):
        torch.manual_seed(0)
        ds = _ds(root, crop_size=128, mode="random", fg_crop_prob=p)
        return np.mean([ds[0]["mask"][3].sum() > 0 for _ in range(60)])

    assert hit_rate(1.0) > 0.95, "biased crops nearly always contain the tiny object"
    assert hit_rate(0.0) < 0.2, "uniform crops on a 1600x1200 image rarely do"


def test_foreground_bias_finds_elongated_objects_whose_bbox_centre_is_off_the_object(tmp_path):
    """Regression: centring on the bbox centre missed thin diagonal / L-shaped objects (seen on real dacl10k
    photos). A thin diagonal stripe across a 2000x1500 image: its bbox centre lies on the stripe only by luck
    for a straight line, so use an L shape, whose bbox centre is in the empty corner."""
    big = _image(2000, 1500)
    l_shape = [[100, 100, 160, 100, 160, 1300, 1900, 1300, 1900, 1360, 100, 1360]]  # long L: vertical bar + horizontal bar
    anns = [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": l_shape, "area": 1, "bbox": [100, 100, 1800, 1260], "iscrowd": 0}]
    root = _write(tmp_path, img=big, anns=anns)
    # the bbox centre (1000, 730) is far from both bars
    torch.manual_seed(0)
    ds = _ds(root, crop_size=128, mode="random", fg_crop_prob=1.0)
    hits = np.mean([ds[0]["mask"][3].sum() > 0 for _ in range(80)])
    assert hits > 0.95, f"only {hits:.0%} of biased crops contain the object"


def test_scale_jitter_stays_within_bounds(tmp_path):
    root = _write(tmp_path)
    ds = _ds(root, crop_size=64, mode="random", scale=0.5, scale_jitter=0.25)
    seen = []
    orig = ds.render
    ds.render = lambda image_id, ox, oy, scale: (seen.append(scale), orig(image_id, ox, oy, scale))[1]
    for _ in range(80):
        ds[0]
    assert min(seen) >= 0.5 / 1.25 - 1e-9 and max(seen) <= 0.5 * 1.25 + 1e-9 and np.std(seen) > 0.01


# ------------------------------------------------------------------------------------- safety / integration
def test_mismatch_between_decoded_image_and_annotation_size_raises(tmp_path):
    root = _write(tmp_path, coco_wh=(300, 400))  # annotations claim a transposed image
    with pytest.raises(ValueError, match="misaligned"):
        _ds(root)[0]


def test_bad_arguments_rejected(tmp_path):
    root = _write(tmp_path)
    with pytest.raises(ValueError):
        _ds(root, mode="sliding")
    with pytest.raises(ValueError):
        _ds(root, scale=0)


def test_item_format_matches_the_whole_image_dataset(tmp_path):
    root = _write(tmp_path)
    crop_item = _ds(root, crop_size=64)[0]
    whole_item = CocoMultiLabelSegDataset(root, root / "coco.json", root / "map.yaml", MVP)[0]
    assert set(crop_item) == set(whole_item)
    assert crop_item["mask"].shape == (5, 64, 64) and crop_item["validity_mask"].shape == (1, 64, 64) and crop_item["image"].shape == (3, 64, 64)
    assert crop_item["source_dataset"] == whole_item["source_dataset"]


@pytest.mark.parametrize("role,expected", [("train", "random"), ("val", "fixed"), ("test", "grid")])
def test_builder_maps_roles_to_crop_modes(tmp_path, role, expected):
    root = _write(tmp_path)
    cfg = DataConfig(resize_mode="native_crop", pixel_scale=0.75, train_crops_per_image=3, train_scale_jitter=0.2, fg_crop_prob=0.4)
    ds = _make_dataset(role, str(root), str(root / "coco.json"), str(root / "map.yaml"), None, None, "src", cfg, val_mode="fixed")
    assert isinstance(ds, CocoCropDataset) and ds.mode == expected and ds.scale == 0.75 and ds.crop_size == 512
    assert ds.fg_crop_prob == 0.4
    assert (ds.scale_jitter, ds.crops_per_image) == ((0.2, 3) if role == "train" else (0.0, 1))


def test_builder_val_mode_grid_and_whole_image_default(tmp_path):
    root = _write(tmp_path)
    ds = _make_dataset("val", str(root), str(root / "coco.json"), str(root / "map.yaml"), None, None, "s", DataConfig(resize_mode="native_crop"), val_mode="grid")
    assert ds.mode == "grid"
    whole = _make_dataset("train", str(root), str(root / "coco.json"), str(root / "map.yaml"), None, None, "s", DataConfig(), val_mode="fixed")
    assert type(whole) is CocoMultiLabelSegDataset


def test_config_validation_of_new_fields():
    for bad in ({"resize_mode": "tiles"}, {"pixel_scale": 0}, {"train_scale_jitter": -0.1}, {"fg_crop_prob": 1.5},
                {"train_crops_per_image": 0}, {"val_crops_per_image": 0}):
        with pytest.raises(ValueError):
            DataConfig(**bad)
    assert DataConfig().resize_mode == "whole_image", "default keeps the paper-recipe behaviour"
