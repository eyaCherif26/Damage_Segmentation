import pytest
import torch
import torch.nn as nn
import torchvision
from segmentation_models_pytorch.encoders import encoders as SMP_ENCODER_REGISTRY

import models
from configs.schema import ModelConfig
from models.bikit_adapter import CoverageError, load_bikit_encoder_into
from models.torchvision_encoders import ENCODER_NAME, OUT_CHANNELS, OUT_INDEXES, TorchvisionMobileNetV3LargeEncoder


def _encoder(pretrained=False, depth=5):
    return TorchvisionMobileNetV3LargeEncoder(OUT_CHANNELS, depth=depth, imagenet_pretrained=pretrained)


def _fake_bikit_checkpoint(features: nn.Module, tmp_path, num_class=6):
    """bikit stores the backbone as nn.Sequential(features, avgpool) under `base.`, so the
    features module ends up one index deep: base.0.<key>."""
    # the real checkpoints hold exactly the features tensors: bikit's base.1 is an
    # AdaptiveAvgPool2d, which has no parameters at all.
    state = {f"base.0.{k}": v for k, v in features.state_dict().items()}
    state.update({"classifier.0.weight": torch.zeros(num_class, 960), "classifier.0.bias": torch.zeros(num_class)})
    path = tmp_path / "bikit_mobilenet.pth"
    torch.save({"base": "mobilenet", "resolution": 512, "num_class": num_class, "state_dict": state}, path)
    return path


def test_encoder_is_registered_for_smp():
    assert ENCODER_NAME in SMP_ENCODER_REGISTRY
    assert SMP_ENCODER_REGISTRY[ENCODER_NAME]["params"]["out_channels"] == OUT_CHANNELS


def test_feature_pyramid_shapes_follow_the_smp_contract():
    enc = _encoder()
    feats = enc(torch.randn(2, 3, 256, 256))
    assert len(feats) == 6
    assert [f.shape[1] for f in feats] == OUT_CHANNELS
    assert [f.shape[-1] for f in feats] == [256, 128, 64, 32, 16, 8], "one feature per stride, highest resolution first"
    assert list(enc.out_channels) == OUT_CHANNELS


@pytest.mark.parametrize("depth", [3, 4, 5])
def test_shallower_depths_return_fewer_features(depth):
    feats = _encoder(depth=depth)(torch.randn(1, 3, 128, 128))
    assert len(feats) == depth + 1


def test_bad_depth_rejected():
    with pytest.raises(ValueError):
        TorchvisionMobileNetV3LargeEncoder(OUT_CHANNELS, depth=6)


def test_stage_outputs_equal_the_reference_torchvision_backbone(tmp_path):
    """The decisive check: identical activations to torchvision's own features module, so the
    stage indices (and later the checkpoint mapping) address the right layers."""
    torch.manual_seed(0)
    reference = torchvision.models.mobilenet_v3_large().features.eval()
    enc = _encoder().eval()
    enc.features.load_state_dict(reference.state_dict())

    x = torch.randn(2, 3, 192, 192)
    with torch.no_grad():
        ours = enc(x)
        wanted, r = {}, x
        for i, layer in enumerate(reference):
            r = layer(r)
            if i in OUT_INDEXES:
                wanted[i] = r.clone()
    for j, i in enumerate(OUT_INDEXES):
        assert torch.equal(ours[j + 1], wanted[i]), f"stage at feature index {i} differs"


def test_bikit_checkpoint_maps_onto_every_encoder_tensor(tmp_path):
    torch.manual_seed(1)
    source = torchvision.models.mobilenet_v3_large().features
    ckpt = _fake_bikit_checkpoint(source, tmp_path)

    model = models.build_model(ModelConfig(name="smp_mobilenet", num_classes=5, dropout=0.0))
    report = model.load_task_pretrained(str(ckpt))
    assert report["coverage"] == 1.0, report
    assert report["matched_tensors"] == len(source.state_dict())

    for k, v in source.state_dict().items():
        assert torch.equal(model.net.encoder.features.state_dict()[k], v), f"{k} not transferred"


def test_classifier_weights_are_not_transferred(tmp_path):
    source = torchvision.models.mobilenet_v3_large().features
    ckpt = _fake_bikit_checkpoint(source, tmp_path)
    model = models.build_model(ModelConfig(name="smp_mobilenet", num_classes=5, dropout=0.0))
    head_before = [p.clone() for p in model.net.segmentation_head.parameters()]
    model.load_task_pretrained(str(ckpt))
    assert all(torch.equal(a, b) for a, b in zip(head_before, model.net.segmentation_head.parameters()))


def test_mismatched_backbone_still_raises(tmp_path):
    """A non-MobileNet bikit checkpoint must not be silently accepted just because rewrites exist."""
    state = {"base.0._conv_stem.weight": torch.zeros(32, 3, 3, 3), "base.0._bn0.weight": torch.zeros(32)}
    path = tmp_path / "wrong.pth"
    torch.save({"base": "efficientnet", "state_dict": state}, path)
    model = models.build_model(ModelConfig(name="smp_mobilenet", num_classes=5, dropout=0.0))
    with pytest.raises(CoverageError):
        model.load_task_pretrained(str(path))


def test_key_rewrites_are_opt_in(tmp_path):
    """Without the rewrite the same checkpoint does not match — this is what used to block MobileNet."""
    source = torchvision.models.mobilenet_v3_large().features
    ckpt = _fake_bikit_checkpoint(source, tmp_path)
    model = models.build_model(ModelConfig(name="smp_mobilenet", num_classes=5, dropout=0.0))
    with pytest.raises(CoverageError):
        load_bikit_encoder_into(model, str(ckpt))  # no key_rewrites


def _only_head_dropout_stochastic(model):
    """Encoder/decoder in eval mode, only the head dropout in train mode. (The native EfficientNet
    encoder has its own stochastic depth in train mode, which would mask what is being tested.)"""
    model.eval()
    model.head_dropout.train()
    return model


@pytest.mark.parametrize("name", ["smp_efficientnet_b0", "smp_mobilenet"])
def test_head_dropout_is_applied_and_isolated(name):
    """smp's get_encoder drops unknown kwargs, so `model.dropout` has to be applied by our wrapper."""
    x = torch.randn(2, 3, 64, 64)

    with_dropout = _only_head_dropout_stochastic(models.build_model(ModelConfig(name=name, num_classes=5, dropout=0.5)))
    torch.manual_seed(0)
    a = with_dropout(x)
    torch.manual_seed(1)
    b = with_dropout(x)
    assert not torch.allclose(a, b), f"{name}: dropout must randomise the output"

    without = _only_head_dropout_stochastic(models.build_model(ModelConfig(name=name, num_classes=5, dropout=0.0)))
    torch.manual_seed(0)
    c = without(x)
    torch.manual_seed(1)
    d = without(x)
    assert torch.allclose(c, d), f"{name}: with dropout=0 nothing else may be stochastic here"

    with_dropout.eval()
    assert torch.allclose(with_dropout(x), with_dropout(x)), f"{name}: eval mode must be deterministic"


@pytest.mark.parametrize("name", ["smp_efficientnet_b0", "smp_mobilenet"])
@pytest.mark.parametrize("decoder", ["fpn", "unet", "deeplabv3plus"])
def test_state_dict_layout_does_not_depend_on_dropout(name, decoder):
    """Regression: wrapping the segmentation head for dropout shifted `segmentation_head.0.*` to
    `.1.*` and broke loading every checkpoint trained before the change. A checkpoint saved with one
    dropout value must load into a model built with any other."""
    with_dropout = models.build_model(ModelConfig(name=name, decoder=decoder, num_classes=5, dropout=0.2))
    without = models.build_model(ModelConfig(name=name, decoder=decoder, num_classes=5, dropout=0.0))
    assert list(with_dropout.state_dict()) == list(without.state_dict())
    assert "net.segmentation_head.0.weight" in with_dropout.state_dict()
    without.load_state_dict(with_dropout.state_dict())
    with_dropout.load_state_dict(without.state_dict())


def test_mobilenet_uses_the_torchvision_encoder():
    m = models.build_model(ModelConfig(name="smp_mobilenet", num_classes=5))
    assert isinstance(m.net.encoder, TorchvisionMobileNetV3LargeEncoder)
    assert m.encoder_id == ENCODER_NAME
