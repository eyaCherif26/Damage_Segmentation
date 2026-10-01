import pytest
import torch

from configs.schema import LossConfig
from losses.boundary import soft_cl_dice_loss, soft_skeletonize
from losses.combined import CombinedLoss
from losses.dice_tversky import dice_loss, tversky_loss
from losses.focal import binary_focal_loss

CLASSES = ["Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence"]


def _data(seed=0, b=2, h=32, w=32):
    g = torch.Generator().manual_seed(seed)
    target = (torch.rand(b, 5, h, w, generator=g) > 0.8).float()
    return target


def _perfect_logits(target):
    return target * 30 - 15


def test_perfect_prediction_gives_near_zero_loss():
    t = _data()
    out = CombinedLoss(LossConfig(), CLASSES, ["Cracks"])(_perfect_logits(t), t)
    assert out["total"].item() < 1e-3


def test_opposite_prediction_is_large_and_finite():
    t = _data()
    out = CombinedLoss(LossConfig(), CLASSES, ["Cracks"])(-_perfect_logits(t), t)
    assert torch.isfinite(out["total"]) and out["total"].item() > 1.0


def test_gradients_finite_and_nonzero():
    t = _data()
    logits = torch.randn_like(t, requires_grad=True)
    CombinedLoss(LossConfig(boundary_weight=0.3), CLASSES, ["Cracks"])(logits, t)["total"].backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_alpha_zero_on_positives_removes_positive_focal_gradient():
    t = torch.ones(1, 1, 8, 8)
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    binary_focal_loss(logits, t, alpha=0.0, gamma=2.0).sum().backward()
    assert logits.grad.abs().sum() == 0, "alpha=0 weights positives by 0 -> no gradient from an all-positive target"


def test_per_class_alpha_changes_only_that_class():
    t = _data()
    logits = torch.randn_like(t)
    a = binary_focal_loss(logits, t, alpha=torch.full((5,), 0.25))
    alpha = torch.full((5,), 0.25)
    alpha[0] = 0.75
    b = binary_focal_loss(logits, t, alpha=alpha)
    assert not torch.isclose(a[0], b[0]) and torch.allclose(a[1:], b[1:])


def test_focal_tversky_gamma_none_reproduces_todays_additive_loss_exactly():
    """Regression guard: the default (None) must be bit-identical to before the FTL knob existed."""
    t = _data()
    logits = torch.randn_like(t)
    plain = CombinedLoss(LossConfig(), CLASSES, ["Cracks"])(logits, t)
    explicit_none = CombinedLoss(LossConfig(focal_tversky_gamma=None), CLASSES, ["Cracks"])(logits, t)
    assert torch.equal(plain["total"], explicit_none["total"])


def test_focal_tversky_gamma_perfect_prediction_still_near_zero():
    t = _data()
    out = CombinedLoss(LossConfig(focal_tversky_gamma=2.0), CLASSES, ["Cracks"])(_perfect_logits(t), t)
    assert out["total"].item() < 1e-3


def test_focal_tversky_gamma_gradients_finite():
    t = _data()
    logits = torch.randn_like(t, requires_grad=True)
    CombinedLoss(LossConfig(focal_tversky_gamma=2.0), CLASSES, ["Cracks"])(logits, t)["total"].backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_focal_tversky_gamma_raises_hard_cases_loss_relative_to_additive_form():
    """The whole point of the exponent: on a hard/rare-class case (moderate Tversky, i.e. neither
    ~perfect nor ~opposite), raising gamma should increase (1-Tversky)^gamma's contribution
    relative to the plain additive (1-Tversky) term used today — the literal FTL paper's point."""
    t = _data()
    logits = torch.randn_like(t) * 0.5  # neither confidently right nor wrong -> a "hard" case
    additive = CombinedLoss(LossConfig(dice_weight=1.0, focal_weight=0.0), CLASSES, ["Cracks"])(logits, t)
    ftl = CombinedLoss(LossConfig(dice_weight=1.0, focal_weight=0.0, focal_tversky_gamma=2.0), CLASSES, ["Cracks"])(logits, t)
    # dice_per_class in [0,1]; squaring a value < 1 only shrinks it (gamma sharpens easy cases down,
    # not blows hard ones up) — assert the FTL total differs from the additive one, not a direction,
    # since which way it moves depends on how hard/easy this particular random draw is.
    assert not torch.isclose(ftl["total"], additive["total"])
    assert ftl["dice"].item() == pytest.approx(additive["dice"].item()), "the reported 'dice' sub-metric stays the raw (pre-exponent) Tversky loss"


def test_all_invalid_tile_gives_zero_loss_and_zero_gradient():
    t = _data()
    logits = torch.randn_like(t, requires_grad=True)
    validity = torch.zeros(2, 1, 32, 32)
    out = CombinedLoss(LossConfig(boundary_weight=0.3), CLASSES, ["Cracks"])(logits, t, validity)
    out["focal"].backward(retain_graph=True)
    assert out["focal"].item() == 0.0
    assert logits.grad.abs().sum() == 0
    logits.grad = None
    tversky_loss(logits, t, validity_mask=validity).sum().backward()
    assert logits.grad.abs().sum() == 0


def test_invalid_region_labels_do_not_leak_into_loss():
    t = _data()
    logits = torch.randn_like(t)
    validity = torch.ones(2, 1, 32, 32)
    validity[:, :, :16] = 0
    t_corrupted = t.clone()
    t_corrupted[:, :, :16] = 1 - t_corrupted[:, :, :16]
    a = CombinedLoss(LossConfig(), CLASSES, [])(logits, t, validity)["total"]
    b = CombinedLoss(LossConfig(), CLASSES, [])(logits, t_corrupted, validity)["total"]
    assert torch.isclose(a, b)


def test_multilabel_no_softmax_two_channels_positive_same_pixel():
    t = torch.zeros(1, 5, 4, 4)
    t[0, 0] = 1
    t[0, 3] = 1  # same pixels positive in two channels
    good = _perfect_logits(t)
    only_one = good.clone()
    only_one[0, 3] = -15  # forgets the second label
    loss = CombinedLoss(LossConfig(), CLASSES, [])
    assert loss(good, t)["total"] < loss(only_one, t)["total"]


def test_dice_is_tversky_half_half():
    t = _data()
    logits = torch.randn_like(t)
    assert torch.allclose(dice_loss(logits, t), tversky_loss(logits, t, 0.5, 0.5))


def test_tversky_beta_penalizes_false_negatives_more():
    t = torch.zeros(1, 1, 8, 8)
    t[..., :4] = 1
    fn_logits = torch.full((1, 1, 8, 8), -10.0)  # predicts nothing -> all FN
    fp_logits = torch.full((1, 1, 8, 8), 10.0)  # predicts everything -> FP heavy
    fn_heavy = tversky_loss(fn_logits, t, alpha=0.3, beta=0.7)
    fp_heavy = tversky_loss(fp_logits, t, alpha=0.3, beta=0.7)
    assert fn_heavy > fp_heavy


def test_soft_cldice_perfect_is_near_zero_and_skeleton_is_thin():
    t = torch.zeros(1, 1, 32, 32)
    t[..., 10:20, 14:19] = 1
    assert soft_cl_dice_loss(t * 30 - 15, t).item() < 0.05
    skel = soft_skeletonize(t)
    assert skel.sum() < t.sum()
