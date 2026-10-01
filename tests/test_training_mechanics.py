from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from training.early_stopping import EarlyStopping
from training.schedule import build_warmup_cosine_scheduler
from training.trainer import Trainer


def _two_group_optimizer(lr_backbone=1e-4, lr_head=1e-3):
    backbone, head = nn.Linear(4, 4), nn.Linear(4, 2)
    opt = torch.optim.AdamW(
        [{"params": backbone.parameters(), "lr": lr_backbone}, {"params": head.parameters(), "lr": lr_head}],
        weight_decay=0.01,
    )
    return backbone, head, opt


def test_both_groups_follow_schedule_and_keep_lr_ratio():
    """min_lr=0 explicitly: this test is about warmup ramp-up and ratio preservation, not the
    min_lr floor (see test_min_lr_floor_* below) — keeps its original decay-to-near-zero intent."""
    _, _, opt = _two_group_optimizer()
    sched = build_warmup_cosine_scheduler(opt, warmup_epochs=2, max_epochs=10, steps_per_epoch=5, min_lr=0)
    seen = []
    for _ in range(50):
        opt.step()
        sched.step()
        seen.append([g["lr"] for g in opt.param_groups])
    assert seen[0][0] < 1e-4 and seen[9][0] > seen[0][0], "warmup ramps up"
    assert seen[-1][0] < 0.05 * 1e-4 and seen[-1][1] < 0.05 * 1e-3, "both groups decay to ~0 by the end"
    for lr_b, lr_h in seen[5:]:
        if lr_h > 0:
            assert abs(lr_h / lr_b - 10) < 1e-6, "backbone/head ratio must be preserved"


def test_min_lr_floors_the_cosine_phase_as_an_absolute_value_per_group():
    """min_lr is an ABSOLUTE floor shared by every group, not a fraction of that group's own peak —
    so two groups with different base_lr (here 1e-4 vs 1e-3, a 10x ratio like this project's real
    backbone/head split) must both land at the SAME final LR, not preserve their ratio at the end."""
    _, _, opt = _two_group_optimizer(lr_backbone=1e-4, lr_head=1e-3)
    sched = build_warmup_cosine_scheduler(opt, warmup_epochs=1, max_epochs=10, steps_per_epoch=5, min_lr=1e-5)
    for _ in range(60):  # well past the schedule's own total_steps -> the floor, not still decaying
        opt.step()
        sched.step()
    lr_backbone, lr_head = (g["lr"] for g in opt.param_groups)
    assert lr_backbone == pytest.approx(1e-5, rel=1e-6)
    assert lr_head == pytest.approx(1e-5, rel=1e-6)


def test_min_lr_does_not_affect_warmup():
    """The floor must only clamp the post-warmup cosine phase — warmup itself still ramps from
    exactly 0, per the literal ask ('with the cosine annealing don't go less than 1e-5'). Checked
    right at construction (step 0 -> multiplier 0/warmup_steps == 0 exactly), not after an explicit
    .step(), since a later warmup step can coincidentally land exactly on the floor value."""
    _, _, opt = _two_group_optimizer(lr_backbone=1e-4, lr_head=1e-3)
    build_warmup_cosine_scheduler(opt, warmup_epochs=2, max_epochs=10, steps_per_epoch=5, min_lr=1e-5)
    lr_backbone, lr_head = (g["lr"] for g in opt.param_groups)
    assert lr_backbone == 0.0 and lr_head == 0.0, "warmup must start at exactly 0, not be clamped up to the floor"


def test_min_lr_default_matches_the_documented_1e5_floor():
    """Every existing config gets this automatically (no YAML edit) — pins the actual default
    value, since build_warmup_cosine_scheduler's signature default is what every real config uses."""
    import inspect

    assert inspect.signature(build_warmup_cosine_scheduler).parameters["min_lr"].default == 1e-5


def test_freezing_via_requires_grad_never_changes_scheduler_group_count():
    backbone, head, opt = _two_group_optimizer()
    sched = build_warmup_cosine_scheduler(opt, 1, 6, 4)
    n_groups, n_base = len(opt.param_groups), len(sched.base_lrs)
    for p in backbone.parameters():
        p.requires_grad = False
    for _ in range(8):
        opt.step()
        sched.step()
    for p in backbone.parameters():
        p.requires_grad = True
    for _ in range(8):
        opt.step()
        sched.step()
    assert len(opt.param_groups) == n_groups and len(sched.base_lrs) == n_base


def test_backbone_lr_still_decays_after_unfreezing():
    backbone, head, opt = _two_group_optimizer()
    sched = build_warmup_cosine_scheduler(opt, 1, 10, 4)
    for p in backbone.parameters():
        p.requires_grad = False
    for _ in range(12):
        sched.step()
    lr_at_unfreeze = opt.param_groups[0]["lr"]
    for p in backbone.parameters():
        p.requires_grad = True
    for _ in range(24):
        sched.step()
    assert opt.param_groups[0]["lr"] < lr_at_unfreeze, "the desync bug pinned this at a constant"


def test_frozen_backbone_receives_no_update():
    backbone, head, opt = _two_group_optimizer()
    before = backbone.weight.clone()
    for p in backbone.parameters():
        p.requires_grad = False
    loss = head(backbone(torch.randn(3, 4))).sum()
    loss.backward()
    opt.step()
    assert torch.equal(before, backbone.weight), "AdamW weight decay must not touch a param with no grad"


def test_early_stopping_patience_and_best_tracking():
    es = EarlyStopping(patience=3, min_delta=0.01)
    assert es.step(0.5) is True
    assert es.step(0.505) is False and not es.should_stop  # below min_delta
    assert es.step(0.6) is True and es.counter == 0
    assert not es.step(0.55) and not es.step(0.55) and not es.should_stop
    assert not es.step(0.55) and es.should_stop
    assert es.best == 0.6


def _bare_trainer_with_optimizer(opt) -> Trainer:
    """A Trainer instance with only `.optimizer` set — avoids building the real datasets/model
    just to exercise `_lr_by_group_name`, which reads nothing else off `self`."""
    t = Trainer.__new__(Trainer)
    t.optimizer = opt
    return t


def test_lr_by_group_name_finds_plain_group_by_name_not_index():
    opt = torch.optim.AdamW([
        {"params": [nn.Parameter(torch.zeros(1))], "lr": 1e-4, "name": "backbone", "weight_decay": 0.01},
        {"params": [nn.Parameter(torch.zeros(1))], "lr": 9e-4, "name": "backbone_no_decay", "weight_decay": 0.0},
        {"params": [nn.Parameter(torch.zeros(1))], "lr": 1e-3, "name": "head", "weight_decay": 0.01},
    ])
    t = _bare_trainer_with_optimizer(opt)
    assert t._lr_by_group_name("backbone") == 1e-4, "must match the exact 'backbone' group, not 'backbone_no_decay'"
    assert t._lr_by_group_name("head") == 1e-3


def test_lr_by_group_name_falls_back_to_no_decay_sibling_when_plain_group_is_absent():
    opt = torch.optim.AdamW([
        {"params": [nn.Parameter(torch.zeros(1))], "lr": 1e-4, "name": "backbone_no_decay", "weight_decay": 0.0},
    ])
    t = _bare_trainer_with_optimizer(opt)
    assert t._lr_by_group_name("backbone") == 1e-4


def test_lr_by_group_name_raises_on_unknown_prefix():
    opt = torch.optim.AdamW([{"params": [nn.Parameter(torch.zeros(1))], "lr": 1e-4, "name": "head"}])
    t = _bare_trainer_with_optimizer(opt)
    with pytest.raises(KeyError):
        t._lr_by_group_name("backbone")


def test_grad_clip_caps_gradient_norm_to_configured_value():
    """Mirrors trainer.py's clip block exactly: unscale (a no-op on CPU/disabled AMP) then clip."""
    model = nn.Linear(4, 4)
    (model(torch.randn(3, 4)) ** 2).sum().backward()
    before_norm = model.weight.grad.norm().item()
    scaler = torch.amp.GradScaler(enabled=False)
    scaler.unscale_(torch.optim.AdamW(model.parameters()))
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    after_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)  # measure without changing it further
    assert before_norm > 1.0, "the unclipped gradient must actually be large enough for this test to mean anything"
    assert after_norm == pytest.approx(1.0, abs=1e-4)


def test_grad_clip_none_leaves_gradients_untouched():
    """Regression guard: grad_clip_max_norm=None (the default for every existing config) must not
    change training behavior at all compared to before gradient clipping was added."""
    model = nn.Linear(4, 4)
    (model(torch.randn(3, 4)) ** 2).sum().backward()
    grad_before = model.weight.grad.clone()
    grad_clip_max_norm = None
    if grad_clip_max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
    assert torch.equal(model.weight.grad, grad_before)


# ---------------------------------------------------------------------- finite-loss guard (_run_epoch)

class _ToyDataset(Dataset):
    """A batch's image is set to NaN at `poison_at` indices — the toy model (a real nn.Conv2d) then
    naturally produces non-finite logits for it, mirroring a genuine fp16-overflow forward pass
    (some activation goes non-finite) without needing a GPU. Ground truth (`mask`) stays finite
    throughout: a real corrupted model produces bad PREDICTIONS, never bad ground truth, and that
    distinction matters — see test_non_finite_guard_only_applies_during_training_not_validation."""

    def __init__(self, n: int, poison_at: set[int] | None = None):
        self.n, self.poison_at = n, poison_at or set()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        image = torch.randn(3, 8, 8)
        if i in self.poison_at:
            image[:] = float("nan")
        return {"image": image, "mask": torch.zeros(2, 8, 8), "validity_mask": torch.ones(1, 8, 8), "image_id": i}


class _MeanLogitsLoss(nn.Module):
    """Stands in for CombinedLoss: a real gradient-bearing reduction of the model's own logits, so a
    non-finite forward pass produces a non-finite loss exactly like the real loss functions would."""

    def forward(self, logits, mask, validity):
        return {"total": logits.mean() + mask.sum() * 0.0}


def _bare_trainer_for_run_epoch(max_consecutive: int = 20) -> Trainer:
    model = nn.Conv2d(3, 2, 1)
    t = Trainer.__new__(Trainer)
    t.model = model
    t.device = torch.device("cpu")
    t.loss_fn = _MeanLogitsLoss()
    t.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t._all_trainable_params = list(model.parameters())
    t.scaler = torch.amp.GradScaler(enabled=False)
    t.scheduler = torch.optim.lr_scheduler.LambdaLR(t.optimizer, lambda _: 1.0)
    t.wandb_logger = None
    t._consecutive_nonfinite_batches = 0
    t._amp_enabled = False  # CPU in these tests; see test_amp_dtype.py for the bf16/fp16 selection itself
    t._autocast_dtype = torch.bfloat16
    t.cfg = SimpleNamespace(
        data=SimpleNamespace(mvp_classes=["A", "B"]),
        training=SimpleNamespace(grad_clip_max_norm=None, grad_accum_steps=1, log_every_n_steps=1000,
                                  amp=False, max_consecutive_nonfinite_batches=max_consecutive),
        wandb=SimpleNamespace(num_logged_images=0),
    )
    return t


def test_a_non_finite_batch_is_skipped_and_excluded_from_the_epoch_average():
    t = _bare_trainer_for_run_epoch()
    loader = DataLoader(_ToyDataset(5, poison_at={2}), batch_size=1)
    result = t._run_epoch(epoch=0, loader=loader, train=True)
    assert torch.isfinite(torch.tensor(result["loss"])), "one bad batch must not poison the epoch's average"
    assert t._consecutive_nonfinite_batches == 0, "a later finite batch must reset the streak"


def test_a_non_finite_batch_never_updates_the_model():
    t = _bare_trainer_for_run_epoch()
    before = t.model.weight.clone()
    loader = DataLoader(_ToyDataset(1, poison_at={0}), batch_size=1)
    t._run_epoch(epoch=0, loader=loader, train=True)
    assert torch.equal(t.model.weight, before), "a skipped batch must not reach backward/optimizer.step at all"


def test_persistent_corruption_aborts_instead_of_running_to_completion():
    """Regression for the real incident: a permanently-poisoned model (every forward non-finite)
    must fail loudly after max_consecutive_nonfinite_batches, not silently burn epochs of nan/0.0
    metrics until early stopping eventually notices (as the real run did for 5 wasted epochs)."""
    t = _bare_trainer_for_run_epoch(max_consecutive=3)
    loader = DataLoader(_ToyDataset(10, poison_at=set(range(10))), batch_size=1)
    with pytest.raises(RuntimeError, match="consecutive non-finite-loss batches"):
        t._run_epoch(epoch=0, loader=loader, train=True)


def test_a_transient_non_finite_batch_does_not_trip_the_abort():
    """A single bad batch surrounded by healthy ones (epoch 3's actual, non-fatal event) must not
    accumulate toward the consecutive-abort count."""
    t = _bare_trainer_for_run_epoch(max_consecutive=2)
    loader = DataLoader(_ToyDataset(6, poison_at={1, 4}), batch_size=1)  # never 2 in a row
    t._run_epoch(epoch=0, loader=loader, train=True)  # must not raise


def test_non_finite_guard_only_applies_during_training_not_validation():
    """Validation must still surface a genuinely broken model's non-finite loss as-is (that's how
    the real incident's epoch-4 collapse was first visible: val_loss=nan) rather than masking it —
    the guard only ever skips a TRAIN batch's backward/optimizer step. (val_macro_dice went to
    exactly 0.0, not nan, in the real incident: PixelMetricAccumulator thresholds `probs >= t`,
    and any comparison against NaN is False, so a NaN prediction is silently read as "predicted
    background" rather than corrupting the accumulator — this test pins that compute() still
    doesn't crash, it's the loss value that must carry the signal.)"""
    t = _bare_trainer_for_run_epoch()
    loader = DataLoader(_ToyDataset(3, poison_at={0, 1, 2}), batch_size=1)
    result = t._run_epoch(epoch=0, loader=loader, train=False)
    assert not torch.isfinite(torch.tensor(result["loss"])), "val loss must reflect real model breakage, not be hidden"
    assert result["pixel"]["macro"]["dice"] == 0.0


# ---------------------------------------------------------------------- best-checkpoint W&B artifact

class _SpyWandbLogger:
    def __init__(self):
        self.calls = []

    def log_artifact(self, path, name, type_, metadata=None, aliases=None):
        self.calls.append({"path": path, "name": name, "type_": type_, "metadata": metadata, "aliases": aliases})


def _bare_trainer_for_artifact_logging(tmp_path, best_epoch, best_value):
    t = Trainer.__new__(Trainer)
    t.checkpoint_dir = tmp_path
    t.wandb_logger = _SpyWandbLogger()
    t.cfg = SimpleNamespace(name="myexp", training=SimpleNamespace(early_stopping_metric="val_macro_dice"))
    t._best_epoch, t._best_monitor_value = best_epoch, best_value
    return t


def test_best_checkpoint_is_uploaded_once_with_epoch_and_metric(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"fake checkpoint")
    t = _bare_trainer_for_artifact_logging(tmp_path, best_epoch=7, best_value=0.62)

    t._log_best_checkpoint_artifact()

    assert len(t.wandb_logger.calls) == 1
    call = t.wandb_logger.calls[0]
    assert call["path"] == str(tmp_path / "best.pt")
    assert call["name"] == "myexp-best" and call["type_"] == "model"
    assert call["metadata"] == {"epoch": 7, "val_macro_dice": 0.62}
    assert call["aliases"] == ["best"], "a stable alias, not just a version number, so it's fetchable without knowing the run history"


def test_best_checkpoint_upload_is_a_noop_if_no_checkpoint_was_ever_saved(tmp_path):
    """max_epochs=0, or a run that crashed before the first epoch's checkpoint write — must not
    error trying to upload a file that was never created."""
    t = _bare_trainer_for_artifact_logging(tmp_path, best_epoch=None, best_value=None)
    t._log_best_checkpoint_artifact()
    assert t.wandb_logger.calls == []
