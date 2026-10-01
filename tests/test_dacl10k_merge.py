import json
from pathlib import Path

import pytest

from data.merge_dacl10k import _polygon_area_and_bbox, discover_categories, merge_split


def _write_labelme(ann_dir: Path, img_dir: Path, name: str, shapes: list[dict], w=100, h=80):
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / name).write_bytes(b"x")
    (ann_dir / (name.replace(".jpg", ".json"))).write_text(
        json.dumps({"imageName": name, "imageWidth": w, "imageHeight": h, "shapes": shapes})
    )


def _poly(label, pts):
    return {"label": label, "points": pts, "shape_type": "polygon"}


@pytest.fixture
def fixture_dirs(tmp_path):
    ann, img = tmp_path / "ann", tmp_path / "img"
    _write_labelme(ann, img, "a.jpg", [_poly("Crack", [[0, 0], [10, 0], [10, 10], [0, 10]]), _poly("Rust", [[20, 20], [30, 20], [25, 30]])])
    _write_labelme(ann, img, "b.jpg", [])
    return ann, img


def test_area_and_bbox_of_square_and_triangle():
    area, bbox = _polygon_area_and_bbox([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert area == 100 and bbox == [0, 0, 10, 10]
    area, bbox = _polygon_area_and_bbox([[20, 20], [30, 20], [25, 30]])
    assert area == 50 and bbox == [20, 20, 10, 10]


def test_categories_are_sorted_union(fixture_dirs):
    ann, _ = fixture_dirs
    assert discover_categories([ann]) == ["Crack", "Rust"]


def test_merge_split_counts_ids_and_schema(fixture_dirs):
    ann, img = fixture_dirs
    coco, report = merge_split(ann, img, {"Crack": 1, "Rust": 2}, "train")
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 2
    assert {c["id"] for c in coco["categories"]} == {1, 2}, "category ids must be 1-based"
    a0 = coco["annotations"][0]
    assert a0["category_id"] == 1 and a0["area"] == 100 and a0["iscrowd"] == 0
    assert a0["segmentation"] == [[0, 0, 10, 0, 10, 10, 0, 10]]
    assert report["zero_annotation_images"] == 1
    assert report["per_class_instance_counts"] == {"Crack": 1, "Rust": 1}


def test_degenerate_polygon_dropped_and_counted(tmp_path):
    ann, img = tmp_path / "ann", tmp_path / "img"
    _write_labelme(ann, img, "c.jpg", [_poly("Crack", [[0, 0], [5, 5]]), _poly("Crack", [[0, 0], [5, 0], [10, 0]])])
    coco, report = merge_split(ann, img, {"Crack": 1}, "train")
    assert len(coco["annotations"]) == 0 and report["degenerate_polygons_dropped"] == 2


def test_non_polygon_shape_raises(tmp_path):
    ann, img = tmp_path / "ann", tmp_path / "img"
    _write_labelme(ann, img, "d.jpg", [{"label": "Crack", "points": [[0, 0], [1, 1]], "shape_type": "rectangle"}])
    with pytest.raises(NotImplementedError):
        merge_split(ann, img, {"Crack": 1}, "train")


def test_missing_image_raises(tmp_path):
    ann, img = tmp_path / "ann", tmp_path / "img"
    _write_labelme(ann, img, "e.jpg", [])
    (img / "e.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        merge_split(ann, img, {"Crack": 1}, "train")
