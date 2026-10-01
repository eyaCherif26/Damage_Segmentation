import numpy as np

from data.validity_mask import apply_validity, no_data_validity_mask, training_validity_mask


def _img(h=120, w=160, value=150):
    return np.full((h, w, 3), value, np.uint8)


def test_training_mask_is_all_valid():
    m = training_validity_mask(10, 20)
    assert m.shape == (1, 10, 20) and m.min() == 1


def test_black_border_is_no_data_and_interior_is_valid():
    img = _img()
    img[:, :30] = 0
    m = no_data_validity_mask(img)
    assert m.shape == (1, 120, 160) and m.dtype == np.float32
    assert m[0, :, :30].max() == 0 and m[0, :, 30:].min() == 1


def test_compression_noise_near_border_still_counts_as_no_data():
    img = _img()
    rng = np.random.default_rng(0)
    img[:, :30] = rng.integers(0, 4, (120, 30, 3))  # noise within tolerance=3
    assert no_data_validity_mask(img)[0, :, :30].max() == 0


def test_isolated_dark_pixels_and_small_shadow_specks_stay_valid():
    img = _img()
    img[10, 10] = (2, 3, 1)
    img[50:53, 60:63] = 0  # 9 px speck < min_region_px
    m = no_data_validity_mask(img)
    assert m.min() == 1


def test_thin_seam_line_is_no_data_but_dark_only_in_one_channel_is_not():
    img = _img()
    img[:, 80:82] = 0  # 2 px wide seam across the whole height
    assert no_data_validity_mask(img)[0, :, 80:82].max() == 0
    img2 = _img()
    img2[:, 40:60, 0] = 0  # only red is dark -> not no-data
    assert no_data_validity_mask(img2).min() == 1


def test_bright_image_and_grayscale_input():
    assert no_data_validity_mask(_img()).min() == 1
    gray = np.full((50, 50), 100, np.uint8)
    gray[:, :20] = 0
    assert no_data_validity_mask(gray)[0, :, :20].max() == 0


def test_apply_validity_broadcasts_over_channels():
    t = np.ones((5, 4, 4), np.float32)
    v = np.ones((1, 4, 4), np.float32)
    v[0, :2] = 0
    out = apply_validity(t, v)
    assert out[:, :2].sum() == 0 and out[:, 2:].min() == 1


def test_subsampled_detection_matches_full_resolution_closely_and_keeps_shape():
    img = np.full((1001, 1503, 3), 120, np.uint8)  # odd sizes: exercises the crop after repeat
    img[:, :250] = 0
    img[700:, :] = 0
    full = no_data_validity_mask(img)
    fast = no_data_validity_mask(img, max_pixels=100_000)
    assert fast.shape == full.shape == (1, 1001, 1503) and fast.dtype == np.float32
    assert (fast != full).mean() < 0.005
    assert fast[0, :, :240].max() == 0 and fast[0, :690, 260:].min() == 1


def test_subsampling_is_skipped_for_small_images():
    img = np.full((100, 100, 3), 120, np.uint8)
    img[:, :30] = 0
    assert np.array_equal(no_data_validity_mask(img), no_data_validity_mask(img, max_pixels=10**6))
