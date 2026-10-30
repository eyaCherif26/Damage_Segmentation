import numpy as np

from inference.noise_filtering import (
    clean_binary_mask,
    component_mean_width,
    filter_by_shape,
    hysteresis_mask,
    link_nearby_fragments,
    morphological_cleanup,
    prune_short_skeleton_spurs,
)


def test_morphological_cleanup_opening_removes_speck_but_not_two_thick_blocks():
    mask = np.zeros((20, 20), dtype=bool)
    mask[1, 1] = True  # isolated 1px speck
    mask[5:15, 2:7] = True  # block A, 10x5 -- thick enough to survive a 3x3 erosion
    mask[5:15, 9:14] = True  # block B, 10x5, separated from A by a 2px gap (cols 7-8)

    out = morphological_cleanup(mask, close_px=0, open_px=1)
    assert not out[1, 1]  # speck removed
    assert out[5:15, 2:7].all() and out[5:15, 9:14].all()  # rectangles exactly restored
    assert not out[5:15, 7:9].any()  # gap not bridged (no closing requested)
    assert out.sum() == 10 * 5 * 2


def test_morphological_cleanup_closing_bridges_a_two_pixel_gap():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 2:7] = True
    mask[5:15, 9:14] = True

    out = morphological_cleanup(mask, close_px=1, open_px=0)
    # closing with a 3x3 SE (radius 1) can bridge a gap of width <= 2*radius = 2
    assert out[5:15, 2:14].all()  # gap filled, outer extent unchanged
    assert out.sum() == 10 * 12


def test_morphological_cleanup_is_a_no_op_at_zero():
    mask = np.zeros((10, 10), dtype=bool)
    mask[3, 3] = True
    out = morphological_cleanup(mask, close_px=0, open_px=0)
    assert np.array_equal(out, mask)


def test_hysteresis_mask_keeps_connected_tail_drops_disconnected_blob():
    prob = np.zeros((20, 20), dtype=np.float32)
    prob[2:5, 2:5] = 0.8  # seed, 3x3, well above high=0.6
    prob[3, 5:8] = 0.3  # faint tail, touching the seed's right edge (col 4), above low=0.2
    prob[10:12, 10:12] = 0.3  # separate faint blob, not touching the seed anywhere

    out = hysteresis_mask(prob, low=0.2, high=0.6)
    assert out[2:5, 2:5].all()
    assert out[3, 5] and out[3, 6] and out[3, 7]
    assert not out[10:12, 10:12].any()
    assert out.sum() == 9 + 3


def test_hysteresis_mask_rejects_low_greater_than_high():
    prob = np.zeros((5, 5), dtype=np.float32)
    try:
        hysteresis_mask(prob, low=0.6, high=0.2)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_hysteresis_mask_empty_seed_returns_empty():
    prob = np.full((5, 5), 0.3, dtype=np.float32)  # everything above low, nothing above high
    out = hysteresis_mask(prob, low=0.2, high=0.6)
    assert not out.any()


def test_prune_short_skeleton_spurs_removes_short_whisker_not_the_main_line():
    skel = np.zeros((10, 20), dtype=bool)
    skel[5, 0:20] = True  # a clean 20px straight line
    skel[1:5, 10] = True  # a 4px whisker hanging off the line at column 10 (rows 1-4)

    # The pixel where the whisker meets the line (row 4, col 10) is 8-adjacent to THREE line
    # pixels (row 5, cols 9/10/11) as well as the whisker above it, so it has degree 4 -- a real
    # junction by this function's own definition, and (by design) never removed, since removing it
    # would poke a hole in the line itself rather than just trimming the spur. The walked path from
    # the whisker tip (row 1, degree 1) to that junction is 4 pixels long (rows 1,2,3,4).
    expected_after_prune = np.zeros_like(skel)
    expected_after_prune[5, 0:20] = True
    expected_after_prune[4, 10] = True  # the junction pixel survives

    pruned = prune_short_skeleton_spurs(skel, min_branch_px=5)
    assert np.array_equal(pruned, expected_after_prune)
    assert pruned.sum() == 21

    # min_branch_px=4 does not prune (4 is not < 4) -- unchanged.
    unpruned = prune_short_skeleton_spurs(skel, min_branch_px=4)
    assert np.array_equal(unpruned, skel)

    # min_branch_px=0 is an explicit no-op regardless.
    noop = prune_short_skeleton_spurs(skel, min_branch_px=0)
    assert np.array_equal(noop, skel)


def test_component_mean_width_of_a_clean_one_pixel_line_is_one():
    crop = np.zeros((5, 20), dtype=bool)
    crop[2, :] = True  # already 1px wide everywhere -- skeletonize is an identity on it
    assert component_mean_width(crop) == 1.0


def test_component_mean_width_spur_pruning_increases_the_measured_width():
    # a 3px-thick, 20px-long bar (true width 3) with a thin 1px whisker sticking off one side --
    # the whisker lengthens the skeleton more than it adds area, understating the true width unless
    # pruned first.
    crop = np.zeros((15, 20), dtype=bool)
    crop[4:7, 0:20] = True  # bar, 3 rows x 20 cols
    crop[7:11, 10] = True  # whisker, 4px, hanging off the bottom edge at column 10

    width_unpruned = component_mean_width(crop, spur_prune_px=0)
    width_pruned = component_mean_width(crop, spur_prune_px=6)
    assert width_pruned > width_unpruned
    assert width_pruned >= 2.5  # pruned width should be close to the bar's true width of 3


def test_link_nearby_fragments_merges_within_gap_leaves_separate_otherwise():
    from skimage.measure import label

    mask = np.zeros((10, 30), dtype=bool)
    mask[5, 0:10] = True  # segment A, endpoints at col 0 and col 9
    mask[5, 13:23] = True  # segment B, endpoints at col 13 and col 22 -- 4px from A's col-9 endpoint

    merged = link_nearby_fragments(mask, gap_px=5)
    assert label(merged, connectivity=2).max() == 1
    assert merged[5, 9:14].all()  # the bridge fills the gap

    not_merged = link_nearby_fragments(mask, gap_px=3)
    assert label(not_merged, connectivity=2).max() == 2
    assert np.array_equal(not_merged, mask)  # gap_px too small -- unchanged


def test_link_nearby_fragments_is_a_no_op_at_zero():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2, 0:3] = True
    mask[2, 6:9] = True
    out = link_nearby_fragments(mask, gap_px=0)
    assert np.array_equal(out, mask)


def test_filter_by_shape_keeps_thin_line_drops_filled_blob():
    mask = np.zeros((30, 30), dtype=bool)
    mask[2, 0:20] = True  # thin line, width ~1
    mask[10:25, 10:25] = True  # filled 15x15 blob, width much larger

    out = filter_by_shape(mask, max_width_px=4.0)
    assert out.sum() == 20
    assert out[2, 0:20].all()
    assert not out[10:25, 10:25].any()


def test_clean_binary_mask_leaves_untouched_class_byte_identical():
    classes = ["Cracks", "Rust"]
    binary = np.zeros((2, 20, 20), dtype=np.uint8)
    binary[0, 5:15, 5:15] = 1
    binary[1, 2:4, 2:4] = 1
    prob_map = binary.astype(np.float32)
    thresholds = {"Cracks": 0.5, "Rust": 0.5}

    out = clean_binary_mask(
        binary, prob_map, classes, thresholds,
        morph_classes=["Cracks"], morph_close_px=1, morph_open_px=0,
    )
    assert np.array_equal(out[1], binary[1])  # Rust untouched -- not named in any flag


def test_clean_binary_mask_with_no_flags_is_fully_unchanged():
    classes = ["Cracks", "Rust"]
    binary = np.zeros((2, 10, 10), dtype=np.uint8)
    binary[0, 1, 1] = 1
    binary[1, 5, 5] = 1
    prob_map = binary.astype(np.float32)
    thresholds = {"Cracks": 0.5, "Rust": 0.5}

    out = clean_binary_mask(binary, prob_map, classes, thresholds)
    assert np.array_equal(out, binary)
    assert out is not binary  # never mutates/returns the same object


def test_clean_binary_mask_merges_a_fragmented_crack_without_the_shape_filter():
    classes = ["Cracks", "Rust"]
    binary = np.zeros((2, 10, 30), dtype=np.uint8)
    binary[0, 5, 0:10] = 1  # fragment A
    binary[0, 5, 13:23] = 1  # fragment B, 4px from A's nearest endpoint -- same geometry as
    # test_link_nearby_fragments_merges_within_gap_leaves_separate_otherwise
    binary[1, 2, 2:4] = 1  # Rust, must stay untouched
    prob_map = binary.astype(np.float32)
    thresholds = {"Cracks": 0.5, "Rust": 0.5}

    out = clean_binary_mask(
        binary, prob_map, classes, thresholds,
        merge_classes=["Cracks"], merge_gap_px=5,
    )
    assert out[0, 5, 9:14].all()  # the bridge fills the former gap
    assert np.array_equal(out[1], binary[1])  # Rust untouched -- not in merge_classes


def test_clean_binary_mask_chains_morphology_then_hysteresis():
    classes = ["Cracks"]
    prob_map = np.zeros((1, 20, 20), dtype=np.float32)
    prob_map[0, 2:5, 2:5] = 0.8  # seed
    prob_map[0, 3, 5:8] = 0.3  # connected faint tail
    binary = (prob_map >= 0.5).astype(np.uint8)  # only the seed survives the plain high threshold

    out = clean_binary_mask(
        binary, prob_map, classes, {"Cracks": 0.5},
        hysteresis_low_thresholds={"Cracks": 0.2},
    )
    assert out[0, 3, 5] and out[0, 3, 6] and out[0, 3, 7]  # tail recovered by hysteresis
