"""Boundary-aware metrics: hard clDice (thin-structure classes, primary) + boundary-F1 (all
classes, cheap secondary metric) — see CLAUDE.md / docs/methodology.md for why clDice is the
chosen default for thin structures like Cracks."""
from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries


def hard_cl_dice(pred_binary: np.ndarray, target_binary: np.ndarray, eps: float = 1e-6) -> float:
    """pred_binary/target_binary: (H, W) bool/binary arrays. Hard (non-differentiable) clDice
    metric using real morphological skeletonization — used for reporting, not backprop (see
    losses/boundary.py for the differentiable soft version used in training)."""
    if pred_binary.sum() == 0 and target_binary.sum() == 0:
        return 1.0
    pred_skel = skeletonize(pred_binary.astype(bool))
    target_skel = skeletonize(target_binary.astype(bool))
    t_prec = (np.sum(pred_skel & target_binary.astype(bool)) + eps) / (np.sum(pred_skel) + eps)
    t_sens = (np.sum(target_skel & pred_binary.astype(bool)) + eps) / (np.sum(target_skel) + eps)
    return float(2 * t_prec * t_sens / (t_prec + t_sens + eps))


def skeleton_length_and_width(binary_mask: np.ndarray) -> tuple[float, float]:
    """binary_mask: (H, W) bool, one class/one image (GT or prediction). Returns
    (skeleton_length_px, mean_width_px) via the same skeletonize() hard_cl_dice already uses:
    length = skeleton foreground pixel count (a standard medial-axis length proxy), width =
    area / length. For scripts/evaluate.py --tiered_report's Tier 3 thin-class extent metric.
    Deliberately NOT shared with hard_cl_dice's own internal skeletonize call (that would couple
    two independently-tested accumulators for a cheap, bounded win) — (0.0, nan) for an empty mask,
    since width is undefined with no skeleton to divide by."""
    if binary_mask.sum() == 0:
        return 0.0, float("nan")
    skel = skeletonize(binary_mask.astype(bool))
    length = float(skel.sum())
    if length == 0:
        return 0.0, float("nan")
    return length, float(binary_mask.sum()) / length


def boundary_f1(pred_binary: np.ndarray, target_binary: np.ndarray, tolerance_px: int = 2, eps: float = 1e-6) -> float:
    """Boundary F1 with a pixel-distance tolerance: a predicted boundary pixel counts as correct
    if a target boundary pixel lies within `tolerance_px` (dilated boundary overlap), and vice
    versa — tolerates small offsets rather than requiring exact pixel alignment."""
    if pred_binary.sum() == 0 and target_binary.sum() == 0:
        return 1.0
    pred_b = find_boundaries(pred_binary.astype(bool), mode="inner")
    target_b = find_boundaries(target_binary.astype(bool), mode="inner")
    if pred_b.sum() == 0 and target_b.sum() == 0 and pred_binary.any() and target_binary.any():
        return 1.0  # both non-empty yet edge-less: both fill the whole image (e.g. a full-frame "Weathering" mask)
    if pred_b.sum() == 0 or target_b.sum() == 0:
        return 0.0

    from scipy.ndimage import binary_dilation

    struct = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1))
    target_b_dilated = binary_dilation(target_b, structure=struct)
    pred_b_dilated = binary_dilation(pred_b, structure=struct)

    precision = (pred_b & target_b_dilated).sum() / (pred_b.sum() + eps)
    recall = (target_b & pred_b_dilated).sum() / (target_b.sum() + eps)
    return float(2 * precision * recall / (precision + recall + eps))
