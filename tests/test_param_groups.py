"""training/param_groups.py: excluding norm/bias params from weight decay, and the trainer's
resulting optimizer construction / grad-clipping / by-name LR lookup."""
import torch
import torch.nn as nn

from training.param_groups import build_param_groups, no_decay_param_ids


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, bias=True)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(self.conv(x))


class _TinyModel(nn.Module):
    """Mirrors the real models' shape: a backbone (conv+bias, BatchNorm weight+bias) and a head
    (linear with bias, LayerNorm weight+bias) — exactly the two module families the no-decay
    logic must catch."""

    def __init__(self):
        super().__init__()
        self.backbone = _TinyBackbone()
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2, bias=True)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        yield from self.norm.parameters()
        yield from self.head.parameters()


def test_no_decay_ids_cover_every_bias_and_every_norm_weight_but_nothing_else():
    model = _TinyModel()
    ids = no_decay_param_ids(model)
    expected = {id(model.backbone.conv.bias), id(model.backbone.bn.weight), id(model.backbone.bn.bias),
                id(model.norm.weight), id(model.norm.bias), id(model.head.bias)}
    assert ids == expected
    assert id(model.backbone.conv.weight) not in ids, "conv weight is a real weight matrix, must still decay"
    assert id(model.head.weight) not in ids, "linear weight is a real weight matrix, must still decay"


def test_build_param_groups_gives_no_decay_groups_weight_decay_zero():
    model = _TinyModel()
    groups = build_param_groups(model, list(model.backbone_parameters()), list(model.head_parameters()),
                                lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.01)
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {"backbone", "backbone_no_decay", "head", "head_no_decay"}
    assert by_name["backbone"]["weight_decay"] == 0.01 and by_name["backbone_no_decay"]["weight_decay"] == 0.0
    assert by_name["head"]["weight_decay"] == 0.01 and by_name["head_no_decay"]["weight_decay"] == 0.0
    assert by_name["backbone"]["lr"] == 1e-4 and by_name["head"]["lr"] == 1e-3
    assert by_name["backbone_no_decay"]["lr"] == 1e-4 and by_name["head_no_decay"]["lr"] == 1e-3
    # every parameter appears in exactly one group
    all_grouped = [id(p) for g in groups for p in g["params"]]
    all_model = [id(p) for p in model.parameters()]
    assert sorted(all_grouped) == sorted(all_model)


def test_empty_no_decay_group_is_omitted_not_left_empty():
    backbone = nn.Linear(4, 4, bias=False)  # no bias, no norm layer -> nothing goes to no-decay

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone

    groups = build_param_groups(M(), list(backbone.parameters()), [], lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.01)
    assert {g["name"] for g in groups} == {"backbone"}, "backbone_no_decay must not appear as an empty group"


def test_weight_decay_actually_zero_shrinks_no_decay_param_less_than_decay_param():
    """End-to-end: with zero gradient, weight decay alone should not move a no-decay param at all,
    while a decay param still shrinks toward zero (this is the actual behavioral difference the
    fix produces, not just a config-dict detail)."""
    model = _TinyModel()
    groups = build_param_groups(model, list(model.backbone_parameters()), list(model.head_parameters()),
                                lr_backbone=0.1, lr_head=0.1, weight_decay=0.5)
    opt = torch.optim.AdamW(groups)
    bias_before = model.head.bias.clone()
    weight_before = model.head.weight.clone()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)  # zero gradient: any movement is purely from weight decay
    opt.step()
    assert torch.equal(model.head.bias, bias_before), "no-decay param must not move under zero gradient"
    assert not torch.equal(model.head.weight, weight_before), "decay param must still shrink under zero gradient"
