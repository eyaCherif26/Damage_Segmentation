"""Opt-in rule-based cleanup of per-class thresholded prediction masks.

Standalone, pure numpy/scipy/skimage functions -- no retraining, no GPU. Not called from
`inference/postprocess.py` or either inference script; see `scripts/postprocess_predictions.py`,
which applies these to an *existing* inference run's saved output as a separate, repeatable layer
(so each technique can be tried individually on a saved example without re-running the model).

Three independent techniques, composed in `clean_binary_mask` in this order (see its own docstring
for why): morphological cleanup -> hysteresis thresholding -> shape-based filtering. Every stage is
a no-op unless explicitly requested for a given class.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from skimage.morphology import reconstruction, skeletonize

from metrics.instance_metrics import extract_instances


def morphological_cleanup(mask: np.ndarray, close_px: int, open_px: int) -> np.ndarray:
    """mask: (H, W) bool. Binary closing (bridge small gaps) then binary opening (remove isolated
    speckle), each with a square (2*px+1)-wide structuring element (same construction
    metrics/boundary_metrics.py::boundary_f1 already uses for its tolerance radius). Closing is
    applied BEFORE opening, deliberately: closing fixes small pixel-scale breaks inside one real
    component before opening's erosion step gets a chance to sever them permanently; opening first
    risks erasing a real thin fragment that closing would have reconnected. `px <= 0` skips that
    operation; `close_px=open_px=0` returns an unchanged copy."""
    out = mask.astype(bool)
    if close_px > 0:
        struct = np.ones((2 * close_px + 1, 2 * close_px + 1))
        out = binary_closing(out, structure=struct)
    if open_px > 0:
        struct = np.ones((2 * open_px + 1, 2 * open_px + 1))
        out = binary_opening(out, structure=struct)
    return out


def hysteresis_mask(prob: np.ndarray, low: float, high: float, seed_mask: np.ndarray | None = None) -> np.ndarray:
    """prob: (H, W) float one-class probability map. Returns (H, W) bool: every low-threshold
    candidate pixel (prob >= low) that is 8-connected, via true morphological reconstruction by
    dilation (skimage.morphology.reconstruction, Vincent 1993), to at least one seed pixel.
    `seed_mask` (default: prob >= high) lets the caller pass an already morphologically-cleaned
    seed instead of the raw high-threshold mask. Correctly follows branching connectivity (e.g. a
    reticulated "net crack"), unlike a hand-rolled "does this blob touch a seed pixel" check.
    Requires low <= high."""
    if low > high:
        raise ValueError(f"hysteresis_mask: low ({low}) must be <= high ({high})")
    candidate = prob >= low
    seed = (seed_mask if seed_mask is not None else (prob >= high)) & candidate
    if not seed.any():
        return np.zeros_like(candidate, dtype=bool)
    reconstructed = reconstruction(
        seed.astype(np.float64), candidate.astype(np.float64), method="dilation", footprint=np.ones((3, 3))
    )
    return reconstructed > 0


def _degree_map(skel: np.ndarray) -> np.ndarray:
    """Count of 8-connected skeleton neighbors for every pixel (0 off-skeleton). Vectorized via a
    padded roll-and-sum instead of a per-pixel neighbor loop."""
    padded = np.pad(skel.astype(np.uint8), 1)
    deg = np.zeros_like(padded, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            deg += np.roll(np.roll(padded, dy, axis=0), dx, axis=1)
    deg = deg[1:-1, 1:-1].copy()
    deg[~skel.astype(bool)] = 0
    return deg


def prune_short_skeleton_spurs(skeleton: np.ndarray, min_branch_px: int) -> np.ndarray:
    """skeleton: (h, w) bool, one component's own skeletonize() output. Detects endpoints (skeleton
    pixels with exactly one 8-neighbor) and walks each along degree-2 pixels to the nearest junction
    (degree >= 3) or another endpoint; if the walked path (excluding a junction pixel reached at the
    end, which is shared structure and never removed) has fewer than `min_branch_px` pixels, it's
    removed. One pass over each endpoint once -- deliberately simple, not the full Discrete Skeleton
    Evolution algorithm (which prunes by reconstruction contribution, not raw length; noted as
    possible future work). `min_branch_px <= 0` is a no-op."""
    if min_branch_px <= 0 or not skeleton.any():
        return skeleton.copy()
    skel = skeleton.astype(bool).copy()
    deg = _degree_map(skel)
    endpoints = list(zip(*np.nonzero(deg == 1)))

    def neighbors8(y, x):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and skel[ny, nx]:
                    yield ny, nx

    to_remove: set[tuple[int, int]] = set()
    for start in endpoints:
        if start in to_remove:
            continue
        path = [start]
        visited = {start}
        cur = start
        while True:
            if cur != start and deg[cur] != 2:
                break  # reached a junction or another endpoint; stop, cur already in path
            nbrs = [n for n in neighbors8(*cur) if n not in visited]
            if not nbrs:
                break
            cur = nbrs[0]
            path.append(cur)
            visited.add(cur)
        if len(path) < min_branch_px:
            last = path[-1]
            if deg[last] >= 3:  # never remove a junction pixel itself -- it's shared structure
                path = path[:-1]
            to_remove.update(path)

    out = skel.copy()
    for y, x in to_remove:
        out[y, x] = False
    return out


def component_mean_width(crop_mask: np.ndarray, spur_prune_px: int = 0) -> float:
    """crop_mask: (h, w) bool, one instance's own bbox-cropped mask (never a full-image mask --
    reuses extract_instances' own crop convention). Skeletonizes (the same skimage.morphology.
    skeletonize call metrics/boundary_metrics.py::skeleton_length_and_width uses), optionally
    pruning short spurs first so one noisy whisker doesn't skew the ratio, then returns
    area / skeleton_length -- the same formula as skeleton_length_and_width, computed directly here
    rather than through it (that function only skeletonizes its own input, it can't accept a
    precomputed/pruned skeleton; mirrors its own documented choice not to couple with hard_cl_dice's
    skeletonize call for a one-line formula)."""
    area = float(crop_mask.sum())
    if area == 0:
        return float("nan")
    skel = skeletonize(crop_mask.astype(bool))
    if spur_prune_px > 0:
        skel = prune_short_skeleton_spurs(skel, spur_prune_px)
    length = float(skel.sum())
    if length == 0:
        return area  # degenerate (e.g. spur-pruned away entirely) -- fall back to raw area, never NaN
    return area / length


def _skeleton_endpoints(crop: np.ndarray, y0: int, x0: int, spur_prune_px: int) -> list[tuple[int, int]]:
    skel = skeletonize(crop.astype(bool))
    if spur_prune_px > 0:
        skel = prune_short_skeleton_spurs(skel, spur_prune_px)
    deg = _degree_map(skel)
    return [(y + y0, x + x0) for y, x in zip(*np.nonzero(deg == 1))]


def link_nearby_fragments(mask: np.ndarray, gap_px: int, spur_prune_px: int = 0) -> np.ndarray:
    """mask: (H, W) bool, one class. Finds each component's skeleton endpoints (after spur-pruning,
    so a noise whisker's tip isn't mistaken for a real crack end) and draws a 1px bridge between the
    closest pair of endpoints from two DIFFERENT components whenever that pair is within `gap_px`
    Euclidean distance. Deliberately more surgical than a larger `morphological_cleanup` close_px,
    which would thicken a component uniformly in every direction regardless of whether another
    fragment is actually nearby. `gap_px <= 0` is a no-op."""
    if gap_px <= 0:
        return mask.copy()
    instances = extract_instances(mask.astype(bool), min_area=1)
    if len(instances) < 2:
        return mask.astype(bool).copy()

    endpoints_per_instance = [
        _skeleton_endpoints(inst["crop"], inst["bbox"][0], inst["bbox"][1], spur_prune_px) for inst in instances
    ]

    out = mask.astype(np.uint8).copy()
    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            best = None
            for y1, x1 in endpoints_per_instance[i]:
                for y2, x2 in endpoints_per_instance[j]:
                    d = ((y1 - y2) ** 2 + (x1 - x2) ** 2) ** 0.5
                    if d <= gap_px and (best is None or d < best[0]):
                        best = (d, (x1, y1), (x2, y2))
            if best is not None:
                _, p1, p2 = best
                cv2.line(out, p1, p2, 1, 1)
    return out.astype(bool)


def filter_by_shape(mask: np.ndarray, max_width_px: float, spur_prune_px: int = 0, link_gap_px: int = 0) -> np.ndarray:
    """mask: (H, W) bool, one class. If `link_gap_px > 0`, bridges nearby fragments first (see
    `link_nearby_fragments`). Then extracts instances with `min_area=1` -- a deliberate *shape*
    decision, intentionally separate from a final export-time min-area *size* filter, which must
    not be conflated with this one -- and keeps only instances whose `component_mean_width` is
    `<= max_width_px`, OR'd back into a fresh mask via each kept instance's own bbox-cropped
    region."""
    working = mask.astype(bool)
    if link_gap_px > 0:
        working = link_nearby_fragments(working, link_gap_px, spur_prune_px=spur_prune_px)
    out = np.zeros_like(working, dtype=bool)
    for inst in extract_instances(working, min_area=1):
        width = component_mean_width(inst["crop"], spur_prune_px=spur_prune_px)
        if width <= max_width_px:
            y0, x0, y1, x1 = inst["bbox"]
            out[y0:y1, x0:x1] |= inst["crop"]
    return out


def clean_binary_mask(
    binary: np.ndarray,
    prob_map: np.ndarray,
    class_names: list[str],
    thresholds: dict[str, float],
    *,
    morph_classes: list[str] | None = None,
    morph_close_px: int = 0,
    morph_open_px: int = 0,
    hysteresis_low_thresholds: dict[str, float] | None = None,
    merge_classes: list[str] | None = None,
    merge_gap_px: int = 0,
    merge_spur_prune_px: int = 0,
    shape_filter_classes: list[str] | None = None,
    shape_filter_max_width_px: float | None = None,
    shape_filter_spur_prune_px: int = 0,
    shape_filter_link_gap_px: int = 0,
) -> np.ndarray:
    """binary/prob_map: (C, H, W). Per class, in order: `morphological_cleanup` (if the class is in
    `morph_classes`) -> `hysteresis_mask` (if the class has a `hysteresis_low_thresholds` entry,
    seeded from the already-cleaned mask, using `thresholds[name]` as the high/seed value) ->
    `link_nearby_fragments` (if the class is in `merge_classes` -- bridges genuinely separate
    components whose nearest skeleton endpoints are within `merge_gap_px`, distinct from hysteresis:
    hysteresis only recovers pixels the model gave *some* probability to, this bridges a real gap
    where the model predicted nothing at all) -> `filter_by_shape` (if the class is in
    `shape_filter_classes`). A class touched by none of these passes through byte-identical to
    `binary[i]`. Returns a new (C, H, W) array; never mutates `binary`.

    Order rationale: morphology conditions the seed hysteresis grows from (removes a speckle before
    it can anchor a false region; bridges 1-2px breaks before opening's erosion can sever them);
    hysteresis then handles a larger scale of probability-connected gap than closing can; merging
    then bridges genuine zero-probability gaps between fragments that hysteresis structurally cannot
    reach; the shape filter runs last so it judges each component's *final* shape, after any
    reconnection/growth from the earlier stages."""
    morph_classes = set(morph_classes or [])
    hysteresis_low_thresholds = hysteresis_low_thresholds or {}
    merge_classes = set(merge_classes or [])
    shape_filter_classes = set(shape_filter_classes or [])

    out = binary.copy()
    for i, name in enumerate(class_names):
        mask = out[i].astype(bool)
        if name in morph_classes:
            mask = morphological_cleanup(mask, morph_close_px, morph_open_px)
        if name in hysteresis_low_thresholds:
            mask = hysteresis_mask(prob_map[i], hysteresis_low_thresholds[name], thresholds[name], seed_mask=mask)
        if name in merge_classes:
            mask = link_nearby_fragments(mask, merge_gap_px, spur_prune_px=merge_spur_prune_px)
        if name in shape_filter_classes:
            if shape_filter_max_width_px is None:
                raise ValueError("shape_filter_max_width_px must be given when shape_filter_classes is set")
            mask = filter_by_shape(
                mask, shape_filter_max_width_px,
                spur_prune_px=shape_filter_spur_prune_px, link_gap_px=shape_filter_link_gap_px,
            )
        out[i] = mask.astype(out.dtype)
    return out
