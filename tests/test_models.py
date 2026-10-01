import pytest
import torch
import torch.nn as nn

import models
from configs.schema import ModelConfig
from models.bikit_adapter import CoverageError, load_bikit_encoder_into
from models.registry import register_model


@pytest.mark.parametrize("name", ["smp_efficientnet_b0", "smp_mobilenet", "segformer"])
def test_registered_models_honor_full_resolution_logits_contract(name):
    m = models.build_model(ModelConfig(name=name, num_classes=5)).eval()
    x = torch.randn(1, 3, 128, 96)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 5, 128, 96)
    bb, hd = list(m.backbone_parameters()), list(m.head_parameters())
    assert bb and hd
    assert not {id(p) for p in bb} & {id(p) for p in hd}, "backbone/head groups must be disjoint"


def test_unknown_model_and_duplicate_registration_raise():
    with pytest.raises(KeyError):
        models.build_model(ModelConfig(name="nope"))
    with pytest.raises(ValueError):
        register_model("segformer")(lambda cfg: None)


def test_freezing_the_real_backbone_actually_freezes_something():
    m = models.build_model(ModelConfig(name="smp_efficientnet_b0", num_classes=5))
    for p in m.backbone_parameters():
        p.requires_grad = False
    frozen = sum(not p.requires_grad for p in m.parameters())
    trainable = sum(p.requires_grad for p in m.parameters())
    assert frozen > 100 and trainable > 0, "guards against a silent no-op freeze (hasattr-style bug from prior projects)"


class _Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.bn = nn.BatchNorm2d(4)


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Enc()
        self.head = nn.Conv2d(4, 2, 1)


def _write_ckpt(path, state):
    torch.save({"base": "toy", "resolution": 224, "state_dict": state}, path)


def test_bikit_adapter_loads_matching_encoder_and_ignores_classifier(tmp_path):
    src = _Enc()
    state = {f"base.{k}": v for k, v in src.state_dict().items()}
    state["classifier.weight"] = torch.zeros(3, 3)
    _write_ckpt(tmp_path / "c.pth", state)
    target = _Wrapper()
    report = load_bikit_encoder_into(target, str(tmp_path / "c.pth"))
    assert report["coverage"] == 1.0
    assert torch.equal(target.encoder.conv.weight, src.conv.weight)


def test_bikit_adapter_rejects_low_coverage(tmp_path):
    state = {"base.0.0.0.weight": torch.zeros(4, 3, 3, 3), "base.0.0.1.weight": torch.zeros(4), "base.1.weight": torch.zeros(2)}
    _write_ckpt(tmp_path / "c.pth", state)
    with pytest.raises(CoverageError):
        load_bikit_encoder_into(_Wrapper(), str(tmp_path / "c.pth"))


def test_bikit_adapter_rejects_checkpoint_without_encoder(tmp_path):
    _write_ckpt(tmp_path / "c.pth", {"classifier.weight": torch.zeros(1)})
    with pytest.raises(CoverageError):
        load_bikit_encoder_into(_Wrapper(), str(tmp_path / "c.pth"))


@pytest.mark.parametrize("decoder", ["fpn", "unet", "deeplabv3plus"])
def test_every_smp_decoder_builds_and_keeps_full_resolution_contract(decoder):
    m = models.build_model(ModelConfig(name="smp_efficientnet_b0", decoder=decoder, num_classes=5)).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 3, 128, 96)).shape == (1, 5, 128, 96)
    assert list(m.backbone_parameters()) and list(m.head_parameters())


def test_default_decoder_is_fpn_and_bad_decoder_rejected():
    assert ModelConfig().decoder == "fpn"
    with pytest.raises(ValueError):
        ModelConfig(decoder="segnet")
