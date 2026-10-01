import pytest
import torch

from training.amp_config import resolve_amp


def test_bf16_is_the_default_dtype_and_needs_no_scaler():
    settings = resolve_amp(amp=True, amp_dtype="bf16", device_type="cuda", bf16_supported=True)
    assert settings.enabled
    assert settings.autocast_dtype is torch.bfloat16
    assert settings.scaler_enabled is False, "bf16 cannot overflow the way fp16 does, so it needs no loss-scaling"
    assert settings.warning is None


def test_fp16_can_still_be_selected_and_keeps_the_scaler():
    """The old behavior must remain available (`amp_dtype: fp16` in an experiment's YAML), not just
    removed — this project's real fp16 NaN incident should stay reproducible on demand."""
    settings = resolve_amp(amp=True, amp_dtype="fp16", device_type="cuda", bf16_supported=True)
    assert settings.autocast_dtype is torch.float16
    assert settings.scaler_enabled is True


def test_disabled_off_cuda_regardless_of_dtype():
    settings = resolve_amp(amp=True, amp_dtype="bf16", device_type="cpu", bf16_supported=False)
    assert settings.enabled is False and settings.scaler_enabled is False


def test_amp_false_disables_everything_even_on_cuda():
    settings = resolve_amp(amp=False, amp_dtype="fp16", device_type="cuda", bf16_supported=True)
    assert settings.enabled is False and settings.scaler_enabled is False


def test_warns_when_bf16_requested_without_hardware_support():
    settings = resolve_amp(amp=True, amp_dtype="bf16", device_type="cuda", bf16_supported=False)
    assert settings.enabled and settings.warning is not None and "bf16" in settings.warning


def test_no_warning_for_fp16_regardless_of_bf16_support():
    settings = resolve_amp(amp=True, amp_dtype="fp16", device_type="cuda", bf16_supported=False)
    assert settings.warning is None


def test_unknown_dtype_raises():
    with pytest.raises(ValueError, match="amp_dtype"):
        resolve_amp(amp=True, amp_dtype="fp8", device_type="cuda", bf16_supported=True)
