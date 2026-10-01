"""data/merge_dacl10k.py's CLI (main()), run as a subprocess — each split (train/val) is
independently optional (a new, single-batch dataset with no train/val split of its own merges
cleanly to just one split), but at least one must be given, and a split's ann_dir/img_dir must be
given together."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "data/merge_dacl10k.py"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=REPO, capture_output=True, text=True, timeout=60)


def _write_labelme(ann_dir: Path, img_dir: Path, name: str, shapes: list[dict], w=100, h=80):
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / name).write_bytes(b"x")
    (ann_dir / name.replace(".jpg", ".json")).write_text(json.dumps({"imageName": name, "imageWidth": w, "imageHeight": h, "shapes": shapes}))


def _poly(label, pts):
    return {"label": label, "points": pts, "shape_type": "polygon"}


def test_single_split_writes_only_that_split(tmp_path):
    ann, img = tmp_path / "ann", tmp_path / "img"
    _write_labelme(ann, img, "a.jpg", [_poly("Spalling", [[0, 0], [10, 0], [10, 10], [0, 10]])])
    out = tmp_path / "out"

    proc = _run("--train_ann_dir", ann, "--train_img_dir", img, "--output_dir", out)

    assert proc.returncode == 0, proc.stderr
    assert (out / "train" / "annotations.json").exists()
    assert not (out / "val").exists(), "no val split was given — nothing should be written for it"
    coco = json.loads((out / "train" / "annotations.json").read_text())
    assert {c["name"] for c in coco["categories"]} == {"Spalling"}


def test_val_only_also_works(tmp_path):
    ann, img = tmp_path / "ann", tmp_path / "img"
    _write_labelme(ann, img, "a.jpg", [_poly("Efflorescence", [[0, 0], [10, 0], [10, 10], [0, 10]])])
    out = tmp_path / "out"

    proc = _run("--val_ann_dir", ann, "--val_img_dir", img, "--output_dir", out)

    assert proc.returncode == 0, proc.stderr
    assert (out / "val" / "annotations.json").exists()
    assert not (out / "train").exists()


def test_neither_split_given_is_a_clear_error(tmp_path):
    proc = _run("--output_dir", tmp_path / "out")
    assert proc.returncode != 0 and "at least one" in proc.stderr


def test_mismatched_pair_is_a_clear_error(tmp_path):
    ann = tmp_path / "ann"
    ann.mkdir()
    proc = _run("--train_ann_dir", ann, "--output_dir", tmp_path / "out")  # --train_img_dir missing
    assert proc.returncode != 0 and "must be given together" in proc.stderr


def test_both_splits_still_work_together(tmp_path):
    """Regression: existing both-splits usage (dacl10k's own two-split raw layout) is unaffected."""
    train_ann, train_img = tmp_path / "train_ann", tmp_path / "train_img"
    val_ann, val_img = tmp_path / "val_ann", tmp_path / "val_img"
    _write_labelme(train_ann, train_img, "a.jpg", [_poly("Crack", [[0, 0], [10, 0], [10, 10], [0, 10]])])
    _write_labelme(val_ann, val_img, "b.jpg", [_poly("Crack", [[0, 0], [10, 0], [10, 10], [0, 10]])])
    out = tmp_path / "out"

    proc = _run("--train_ann_dir", train_ann, "--train_img_dir", train_img,
                "--val_ann_dir", val_ann, "--val_img_dir", val_img, "--output_dir", out)

    assert proc.returncode == 0, proc.stderr
    assert (out / "train" / "annotations.json").exists() and (out / "val" / "annotations.json").exists()
