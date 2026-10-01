import numpy as np
from pycocotools import mask as mask_utils

from data.common_schema import decode_segmentation, rasterize_annotations

H, W = 20, 30
SQUARE = [[5, 5, 15, 5, 15, 15, 5, 15]]  # x1,y1,... a 10x10 square at x,y in [5,15)


def _ann(cat, seg):
    return {"category_id": cat, "segmentation": seg}


def test_polygon_matches_expected_area():
    m = decode_segmentation(SQUARE, H, W)
    assert m.shape == (H, W) and m.dtype == np.float32
    assert set(np.unique(m)) <= {0.0, 1.0}
    assert 80 <= m.sum() <= 121  # 10x10 square, rasterization edge tolerance
    assert m[10, 10] == 1.0 and m[0, 0] == 0.0


def test_rle_roundtrips_through_same_path():
    poly_mask = decode_segmentation(SQUARE, H, W)
    rle = mask_utils.encode(np.asfortranarray(poly_mask.astype(np.uint8)))
    from_rle = decode_segmentation(rle, H, W)
    assert np.array_equal(poly_mask, from_rle)


def test_uncompressed_rle_dict():
    poly_mask = decode_segmentation(SQUARE, H, W)
    rle = mask_utils.encode(np.asfortranarray(poly_mask.astype(np.uint8)))
    counts = mask_utils.decode(rle)
    # build an uncompressed RLE (column-major run lengths)
    flat = counts.flatten(order="F")
    runs, prev, n = [], 0, 0
    for v in flat:
        if v == prev:
            n += 1
        else:
            runs.append(n)
            prev, n = v, 1
    runs.append(n)
    if flat[0] == 1:
        runs = [0] + runs
    uncompressed = {"size": [H, W], "counts": runs}
    assert np.array_equal(decode_segmentation(uncompressed, H, W), poly_mask)


def test_multilabel_overlap_keeps_both_channels():
    rust_in_spalling = [[8, 8, 12, 8, 12, 12, 8, 12]]
    anns = [_ann(1, SQUARE), _ann(2, rust_in_spalling)]
    mask = rasterize_annotations(anns, H, W, {1: 0, 2: 1}, num_channels=2)
    overlap = np.logical_and(mask[0] > 0, mask[1] > 0)
    assert overlap.sum() > 0, "rust-inside-spalling must be present in BOTH channels"
    assert mask[1].sum() < mask[0].sum()


def test_unmapped_category_contributes_nothing():
    mask = rasterize_annotations([_ann(99, SQUARE)], H, W, {1: 0}, num_channels=1)
    assert mask.sum() == 0


def test_two_annotations_same_channel_are_ored_not_summed():
    anns = [_ann(1, SQUARE), _ann(1, SQUARE)]
    mask = rasterize_annotations(anns, H, W, {1: 0}, num_channels=1)
    assert mask.max() == 1.0


def test_empty_polygon_list():
    assert decode_segmentation([], H, W).sum() == 0
