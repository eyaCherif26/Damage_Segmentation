"""Pre-training sanity checks on a COCO annotation file.

Adapted from a pattern proven useful in a sibling project (`damage-detection-mvp/data/
schema_validator.py`): catches box-only / `segmentation: null` annotation files and label-mapping
coverage gaps *before* a training run starts, rather than discovering them via a confusing
downstream crash or (worse) silently training a mask head against garbage masks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def raise_if_failed(self, context: str = "") -> None:
        if not self.ok:
            prefix = f"{context}: " if context else ""
            raise ValueError(prefix + "schema validation failed:\n" + "\n".join(f"  - {e}" for e in self.errors))


def _is_bbox_rectangle_polygon(polygon: list[float], bbox: list[float]) -> bool:
    """Heuristic: a single 4-point polygon that traces exactly the bbox rectangle is very likely
    a box-only annotation masquerading as a segmentation, not a real polygon mask."""
    if len(polygon) != 8:
        return False
    x, y, w, h = bbox
    corners = {(round(polygon[i], 1), round(polygon[i + 1], 1)) for i in range(0, 8, 2)}
    expected = {
        (round(x, 1), round(y, 1)),
        (round(x + w, 1), round(y, 1)),
        (round(x + w, 1), round(y + h, 1)),
        (round(x, 1), round(y + h, 1)),
    }
    return corners == expected


def validate_coco_file(
    path: str | Path,
    known_category_names: set[str] | None = None,
    max_bbox_rectangle_ratio: float = 0.5,
) -> ValidationReport:
    """Validates a COCO annotation file for the failure modes this project has seen bite prior
    attempts: missing/null segmentation, and (if `known_category_names` is given, e.g. from a
    label mapping) unknown category names."""
    report = ValidationReport()
    with open(path) as f:
        coco = json.load(f)

    for key in ("images", "annotations", "categories"):
        if key not in coco:
            report.errors.append(f"missing top-level key {key!r}")
    if not report.ok:
        return report

    category_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    if 0 in category_id_to_name:
        report.errors.append("category id 0 is present — 0 collides with background conventions in pycocotools/COCOeval; category ids must be 1-based")

    if known_category_names is not None:
        unknown = set(category_id_to_name.values()) - known_category_names
        if unknown:
            report.errors.append(f"category names not covered by the label mapping: {sorted(unknown)}")

    image_ids = {img["id"] for img in coco["images"]}
    n_null_segmentation = 0
    n_bbox_rectangle = 0
    n_missing_image = 0
    for ann in coco["annotations"]:
        if ann.get("image_id") not in image_ids:
            n_missing_image += 1
        seg = ann.get("segmentation")
        if seg is None or seg == [] or seg == {}:
            n_null_segmentation += 1
            continue
        if ann.get("iscrowd", 0) == 0 and isinstance(seg, list) and len(seg) == 1 and "bbox" in ann:
            if _is_bbox_rectangle_polygon(seg[0], ann["bbox"]):
                n_bbox_rectangle += 1

    if n_missing_image:
        report.errors.append(f"{n_missing_image} annotation(s) reference an image_id not present in 'images'")
    if n_null_segmentation:
        report.errors.append(
            f"{n_null_segmentation} annotation(s) have null/empty segmentation — this is the "
            "box-only-labels-masquerading-as-segmentation failure mode documented in CLAUDE.md; "
            "a mask head trained against these wastes capacity at best."
        )
    n_annotations = len(coco["annotations"])
    if n_annotations and n_bbox_rectangle / n_annotations > max_bbox_rectangle_ratio:
        report.warnings.append(
            f"{n_bbox_rectangle}/{n_annotations} annotations ({n_bbox_rectangle / n_annotations:.0%}) "
            "are 4-point polygons exactly tracing their own bbox — likely box-only labels, not real "
            "segmentation masks. Below the failure threshold, but worth a manual spot check."
        )

    return report


def assert_disjoint_image_ids(*id_sets: set[int], names: list[str] | None = None) -> None:
    """Guards against a train/val split carve accidentally leaking images across splits."""
    names = names or [f"split_{i}" for i in range(len(id_sets))]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            overlap = id_sets[i] & id_sets[j]
            if overlap:
                raise ValueError(f"{names[i]!r} and {names[j]!r} share {len(overlap)} image id(s): {sorted(overlap)[:10]}...")
