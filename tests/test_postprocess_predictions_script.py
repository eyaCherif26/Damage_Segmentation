"""End-to-end test of scripts/postprocess_predictions.py as a subprocess, against a hand-built
synthetic --input_dir (prob_<class>.png + run_info.json) -- no model/checkpoint/GPU needed, since
this script only ever reads an already-produced inference run's saved output."""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/postprocess_predictions.py"
CLASSES = ["Cracks", "Rust"]
THRESHOLDS = {"Cracks": 0.5, "Rust": 0.5}


def _make_input_dir(tmp_path) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    h, w = 20, 20

    cracks = np.zeros((h, w), dtype=np.float32)
    cracks[2:5, 2:5] = 0.8  # seed, above the 0.5 threshold
    cracks[3, 5:8] = 0.3  # faint connected tail, below 0.5 but above a low hysteresis threshold
    cracks[18, 18] = 0.6  # isolated speck (far from the seed/tail), above 0.5, for the morphology test

    rust = np.zeros((h, w), dtype=np.float32)
    rust[10:15, 10:15] = 0.9  # a solid blob, untouched by any flag in most sub-tests

    for name, arr in [("Cracks", cracks), ("Rust", rust)]:
        cv2.imwrite(str(d / f"prob_{name}.png"), (arr * 255).astype(np.uint8))

    (d / "run_info.json").write_text(json.dumps({
        "thresholds": THRESHOLDS,
        "class_colors_rgb": {"Cracks": [255, 0, 0], "Rust": [0, 255, 0]},
        "image": None,
    }))
    return d


def _make_fragmented_input_dir(tmp_path) -> Path:
    """A Cracks mask split into two separate line fragments 4px apart (same geometry as
    test_link_nearby_fragments_merges_within_gap_leaves_separate_otherwise), for the merge tests."""
    d = tmp_path / "run_fragmented"
    d.mkdir()
    h, w = 10, 30

    cracks = np.zeros((h, w), dtype=np.float32)
    cracks[5, 0:10] = 0.9  # fragment A
    cracks[5, 13:23] = 0.9  # fragment B

    for name, arr in [("Cracks", cracks), ("Rust", np.zeros((h, w), dtype=np.float32))]:
        cv2.imwrite(str(d / f"prob_{name}.png"), (arr * 255).astype(np.uint8))

    (d / "run_info.json").write_text(json.dumps({
        "thresholds": THRESHOLDS,
        "class_colors_rgb": {"Cracks": [255, 0, 0], "Rust": [0, 255, 0]},
        "image": None,
    }))
    return d


def _run(input_dir, output_dir, *extra, expect_ok=True):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--input_dir", str(input_dir), "--output_dir", str(output_dir), *map(str, extra)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )
    if expect_ok:
        assert proc.returncode == 0, proc.stderr[-2000:]
    return proc


def test_no_flags_at_all_errors_clearly(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    proc = _run(input_dir, tmp_path / "out", expect_ok=False)
    assert proc.returncode != 0
    assert "nothing to do" in proc.stderr


@pytest.mark.parametrize("flags,message_fragment", [
    (["--morph_close_px", "1"], "only apply together with --morph_classes"),
    (["--morph_classes", "Cracks"], "would be a no-op"),
    (["--shape_filter_max_width_px", "4"], "only applies together with --shape_filter_classes"),
    (["--shape_filter_classes", "Cracks"], "requires --shape_filter_max_width_px"),
    (["--shape_filter_classes", "NotAClass", "--shape_filter_max_width_px", "4"], "not in this run's classes"),
    (["--merge_gap_px", "5"], "only applies together with --merge_classes"),
    (["--merge_classes", "Cracks"], "requires --merge_gap_px"),
    (["--merge_classes", "NotAClass", "--merge_gap_px", "5"], "not in this run's classes"),
])
def test_validation_errors(tmp_path, flags, message_fragment):
    input_dir = _make_input_dir(tmp_path)
    proc = _run(input_dir, tmp_path / "out", *flags, expect_ok=False)
    assert proc.returncode != 0
    assert message_fragment in proc.stderr


def test_merge_only_bridges_fragmented_crack(tmp_path):
    input_dir = _make_fragmented_input_dir(tmp_path)
    out = tmp_path / "out"
    _run(input_dir, out, "--merge_classes", "Cracks", "--merge_gap_px", "5", "--min_area", "1")

    coco = json.loads((out / "predictions.coco.json").read_text())
    cat_by_name = {c["name"]: c["id"] for c in coco["categories"]}
    cracks_boxes = [a["bbox"] for a in coco["annotations"] if a["category_id"] == cat_by_name["Cracks"]]
    # the two fragments (cols 0-9 and 13-22) are now bridged into one instance spanning the full range
    assert len(cracks_boxes) == 1
    x, y, bw, bh = cracks_boxes[0]
    assert x == 0 and x + bw == 23


def test_merge_gap_too_small_leaves_fragments_separate(tmp_path):
    input_dir = _make_fragmented_input_dir(tmp_path)
    out = tmp_path / "out"
    _run(input_dir, out, "--merge_classes", "Cracks", "--merge_gap_px", "1", "--min_area", "1")

    coco = json.loads((out / "predictions.coco.json").read_text())
    cat_by_name = {c["name"]: c["id"] for c in coco["categories"]}
    cracks_boxes = [a["bbox"] for a in coco["annotations"] if a["category_id"] == cat_by_name["Cracks"]]
    assert len(cracks_boxes) == 2


def test_hysteresis_only_recovers_connected_tail_leaves_rust_untouched(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    out = tmp_path / "out"
    # --min_area 1: the synthetic seed+tail (12px total) is smaller than the default min_area (16),
    # which would otherwise drop it before this test can inspect its bbox.
    _run(input_dir, out, "--hysteresis_low_threshold", "Cracks=0.2", "--min_area", "1")

    coco = json.loads((out / "predictions.coco.json").read_text())
    cat_by_name = {c["name"]: c["id"] for c in coco["categories"]}
    cracks_boxes = [a["bbox"] for a in coco["annotations"] if a["category_id"] == cat_by_name["Cracks"]]
    # the seed (3x3) and the now-recovered tail are 8-connected -> one merged instance whose bbox
    # spans from the seed's left edge (col 2) to the tail's right edge (col 8); the isolated speck
    # at (18, 18) is untouched by hysteresis and shows up as its own separate tiny instance here
    # since --min_area 1 no longer filters it out -- only asserting the merged box exists.
    assert any(x == 2 and x + bw == 8 for x, y, bw, bh in cracks_boxes)

    info = json.loads((out / "postprocess_info.json").read_text())
    assert info["predicted_pixel_share_after"]["Rust"] == info["predicted_pixel_share_before"]["Rust"]


def test_morph_only_removes_speck(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    out = tmp_path / "out"
    _run(input_dir, out, "--morph_classes", "Cracks", "--morph_open_px", "1")

    info = json.loads((out / "postprocess_info.json").read_text())
    # the isolated 1px speck at (1,1) cannot survive a 3x3 opening, so Cracks' predicted share drops
    assert info["predicted_pixel_share_after"]["Cracks"] < info["predicted_pixel_share_before"]["Cracks"]


def test_shape_filter_keeps_thin_seed_drops_nothing_when_width_is_generous(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    out = tmp_path / "out"
    # the 3x3 seed's width is well under a generous max_width_px -- shape filter should be a
    # practical no-op here, confirming it doesn't wrongly reject a small compact instance
    _run(input_dir, out, "--shape_filter_classes", "Cracks", "--shape_filter_max_width_px", "10")
    info = json.loads((out / "postprocess_info.json").read_text())
    assert info["predicted_pixel_share_after"]["Cracks"] == info["predicted_pixel_share_before"]["Cracks"]


def test_output_files_present_and_input_dir_untouched(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    before = {p.name: p.stat().st_mtime_ns for p in input_dir.iterdir()}
    before_hashes = {p.name: p.read_bytes() for p in input_dir.iterdir()}

    out = tmp_path / "out"
    _run(input_dir, out, "--morph_classes", "Cracks", "--morph_open_px", "1")

    assert (out / "predictions.coco.json").is_file()
    assert (out / "overlay.png").is_file()
    assert (out / "postprocess_info.json").is_file()

    after = {p.name: p.stat().st_mtime_ns for p in input_dir.iterdir()}
    after_hashes = {p.name: p.read_bytes() for p in input_dir.iterdir()}
    assert before == after
    assert before_hashes == after_hashes


def test_default_output_dir_is_input_dir_slash_postprocessed(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--input_dir", str(input_dir),
         "--morph_classes", "Cracks", "--morph_open_px", "1"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (input_dir / "postprocessed" / "predictions.coco.json").is_file()
