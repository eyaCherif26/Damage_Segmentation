import pickle

import pytest
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

import models
from configs.schema import ModelConfig
from models.segformer_pretrained import (
    SegformerCheckpointError,
    extract_state_dict,
    infer_segformer_config,
    load_segformer_pretrained,
)

MIT_B0 = dict(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], decoder_hidden_size=256)


def _tiny_segformer(num_labels: int) -> SegformerForSemanticSegmentation:
    from transformers import SegformerConfig

    cfg = SegformerConfig(num_encoder_blocks=4, depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 20, 32],
                          num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4],
                          decoder_hidden_size=16, num_labels=num_labels)
    return SegformerForSemanticSegmentation(cfg)


def _legacy_keys(new_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename current (5.x) keys to the legacy (4.x) spelling the local checkpoints use."""
    out = {}
    for k, v in new_state.items():
        k = k.replace("segformer.stages.", "segformer.encoder.__STAGE__")
        if "__STAGE__" in k:
            head, rest = k.split("__STAGE__")
            idx, rest = rest.split(".", 1)
            if rest.startswith("blocks."):
                rest = rest[len("blocks.") :]
                j, tail = rest.split(".", 1)
                tail = (tail.replace("attention.q_proj", "attention.self.query").replace("attention.k_proj", "attention.self.key")
                        .replace("attention.v_proj", "attention.self.value").replace("attention.o_proj", "attention.output.dense")
                        .replace("attention.sequence_reduction.sequence_reduction", "attention.self.sr")
                        .replace("attention.sequence_reduction.layer_norm", "attention.self.layer_norm")
                        .replace("layernorm_before", "layer_norm_1").replace("layernorm_after", "layer_norm_2")
                        .replace("mlp.fc1", "mlp.dense1").replace("mlp.fc2", "mlp.dense2"))
                k = f"{head}block.{idx}.{j}.{tail}"
            elif rest.startswith("layer_norm."):
                k = f"{head}layer_norm.{idx}.{rest.split('.', 1)[1]}"
            elif rest.startswith("patch_embeddings."):
                k = f"{head}patch_embeddings.{idx}.{rest.split('.', 1)[1]}"
        k = k.replace("decode_head.linear_projections.", "decode_head.linear_c.")
        out[k] = v
    return out


@pytest.fixture
def legacy_state():
    torch.manual_seed(0)
    return _legacy_keys(_tiny_segformer(19).state_dict())


def test_extract_reads_plain_state_dict(tmp_path, legacy_state):
    p = tmp_path / "sd.pth"
    torch.save(legacy_state, p)
    state, fmt = extract_state_dict(p)
    assert fmt == "state_dict" and len(state) == len(legacy_state)


def test_extract_reads_nested_state_dict_key(tmp_path, legacy_state):
    p = tmp_path / "wrapped.pth"
    torch.save({"state_dict": legacy_state, "epoch": 3}, p)
    state, _ = extract_state_dict(p)
    assert "decode_head.classifier.weight" in state


def test_wrapper_prefix_is_stripped(tmp_path, legacy_state):
    p = tmp_path / "wrapped_prefix.pth"
    torch.save({f"segformer.{k}": v for k, v in legacy_state.items()}, p)
    state, _ = extract_state_dict(p)
    assert "decode_head.classifier.weight" in state
    assert not any(k.startswith("segformer.decode_head") for k in state)


class SegModelForTest(nn.Module):
    """Module-level so pickle can save it; deleted before loading to mimic the real checkpoint,
    whose `__main__.SegModel` / `vit_model_utils.SegModel` class does not exist here."""

    def __init__(self):
        super().__init__()
        self.segformer = _tiny_segformer(19)


def test_extract_reads_pickled_model_without_original_classes(tmp_path, monkeypatch):
    import sys

    obj = SegModelForTest()
    p = tmp_path / "obj.pth"
    torch.save(obj, p)
    monkeypatch.delattr(sys.modules[__name__], "SegModelForTest")  # the class is now unavailable
    with pytest.raises(AttributeError):
        torch.load(p, map_location="cpu", weights_only=False)  # a normal load cannot cope

    state, fmt = extract_state_dict(p)
    assert fmt == "pickled_model"
    assert "decode_head.classifier.weight" in state
    expected = {f"segformer.{k}" for k in obj.segformer.state_dict()}
    assert len(state) == len(expected)
    for k, v in obj.segformer.state_dict().items():
        assert torch.equal(state[k], v), f"{k} not recovered from the pickled object"


def test_config_inferred_from_shapes(legacy_state):
    cfg = infer_segformer_config(legacy_state, num_labels=5)
    assert list(cfg.hidden_sizes) == [8, 16, 20, 32] and list(cfg.depths) == [1, 1, 1, 1]
    assert cfg.num_labels == 5 and cfg.decoder_hidden_size == 16


def test_roundtrip_preserves_every_weight_bit_exactly(tmp_path):
    """The decisive check: legacy-named weights must land on the right parameters — a q/k/v mix-up
    would be shape-compatible but wrong, so compare values tensor by tensor."""
    torch.manual_seed(1)
    source = _tiny_segformer(5)
    legacy = _legacy_keys(source.state_dict())
    p = tmp_path / "src.pth"
    torch.save(legacy, p)

    target = _tiny_segformer(5)
    report = load_segformer_pretrained(target, p, load_decode_head=True, load_classifier=True)
    assert report["unexpected_keys"] == [] and report["classifier_reinitialised"] is False

    src_sd, tgt_sd = source.state_dict(), target.state_dict()
    assert set(src_sd) == set(tgt_sd)
    for k in src_sd:
        assert torch.equal(src_sd[k], tgt_sd[k]), f"{k} did not survive the legacy-key conversion"


def test_outputs_match_the_source_model(tmp_path):
    torch.manual_seed(2)
    source = _tiny_segformer(5).eval()
    p = tmp_path / "src.pth"
    torch.save(_legacy_keys(source.state_dict()), p)
    target = _tiny_segformer(5).eval()
    load_segformer_pretrained(target, p, load_classifier=True)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(source(pixel_values=x).logits, target(pixel_values=x).logits, atol=1e-6)


def test_classifier_is_fresh_by_default_and_class_count_may_differ(tmp_path):
    torch.manual_seed(3)
    source = _tiny_segformer(19)
    p = tmp_path / "src19.pth"
    torch.save(_legacy_keys(source.state_dict()), p)

    target = _tiny_segformer(5)
    before = target.decode_head.classifier.weight.clone()
    report = load_segformer_pretrained(target, p)
    assert report["source_num_labels"] == 19 and report["classifier_reinitialised"] is True
    assert torch.equal(target.decode_head.classifier.weight, before), "our own 5-class head must be kept"
    assert torch.equal(target.decode_head.linear_fuse.weight, source.decode_head.linear_fuse.weight), "decode head still transfers"


def test_encoder_only_when_decode_head_disabled(tmp_path):
    torch.manual_seed(4)
    source = _tiny_segformer(5)
    p = tmp_path / "src.pth"
    torch.save(_legacy_keys(source.state_dict()), p)
    target = _tiny_segformer(5)
    fresh_fuse = target.decode_head.linear_fuse.weight.clone()
    load_segformer_pretrained(target, p, load_decode_head=False)
    assert torch.equal(target.decode_head.linear_fuse.weight, fresh_fuse)
    assert torch.equal(target.segformer.stages[0].patch_embeddings.proj.weight, source.segformer.stages[0].patch_embeddings.proj.weight)


def test_architecture_mismatch_raises(tmp_path, legacy_state):
    p = tmp_path / "small.pth"
    torch.save(legacy_state, p)
    bigger = _tiny_segformer(5)
    bigger.config.hidden_sizes = [32, 64, 160, 256]
    with pytest.raises(SegformerCheckpointError, match="does not match"):
        load_segformer_pretrained(bigger, p)


def test_non_segformer_checkpoint_raises(tmp_path):
    p = tmp_path / "junk.pth"
    torch.save({"conv.weight": torch.zeros(4, 3, 3, 3)}, p)
    with pytest.raises(SegformerCheckpointError):
        load_segformer_pretrained(_tiny_segformer(5), p)


def test_empty_checkpoint_raises(tmp_path):
    p = tmp_path / "empty.pth"
    torch.save({}, p)
    with pytest.raises(SegformerCheckpointError, match="no tensors"):
        extract_state_dict(p)


def test_task_pretrained_is_accepted_for_segformer_in_config():
    cfg = ModelConfig(name="segformer", init_mode="task_pretrained", task_pretrained_checkpoint="x.pth")
    assert cfg.init_mode == "task_pretrained"
    with pytest.raises(ValueError):
        ModelConfig(name="segformer", init_mode="task_pretrained")


def test_every_wrapper_exposes_load_task_pretrained():
    for name in ("segformer", "smp_efficientnet_b0", "smp_mobilenet"):
        m = models.build_model(ModelConfig(name=name, num_classes=5))
        assert callable(getattr(m, "load_task_pretrained", None)), name
