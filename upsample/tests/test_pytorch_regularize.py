"""Tests for sr_pytorch.regularize's new ``strength``/``floor_mode`` params.

Backported from torch_n3's ``tests/test_denoise.py`` (the portable,
property-only half of it — the N3-pipeline-wiring tests there don't apply
here). Distinct from ``tests/test_regularize.py``, which tests ``sr.regularize``
(the compiled C extension) and skips when it isn't built; these tests exercise
``sr_pytorch.regularize`` directly and need no C extension.
"""

import numpy as np
import torch

from sr_pytorch.regularize import regularize


def _sigma(shape, value=5.0):
    return torch.full(shape, value, dtype=torch.float64)


def test_default_args_reproduce_pre_change_output():
    """floor_mode='legacy', strength=1.0 (the defaults) must be a true
    no-op relative to the original regularize(), since tests/test_psnr.py's
    calibrated PSNR floors were measured against it."""
    rng = np.random.default_rng(0)
    vol = torch.from_numpy(rng.random((16, 16, 16))).to(torch.float64) * 256.0
    sigma = torch.from_numpy(rng.random((16, 16, 16))).to(torch.float64) * 10.0

    out_default = regularize(vol, 3, 1, sigma, factors=(2, 2, 2))
    out_explicit = regularize(vol, 3, 1, sigma, factors=(2, 2, 2),
                              strength=1.0, floor_mode="legacy")
    assert torch.equal(out_default, out_explicit)
    # Regression value captured against the pre-change implementation
    # (git-stashed regularize.py) on this exact seed/shape, before this
    # backport's edits were applied -- see the backport plan's verification
    # step. Confirms floor_mode="legacy"/strength=1.0 is a true no-op
    # default, so tests/test_psnr.py's calibrated floors are unaffected.
    assert out_default.sum().item() == 520529.6381982323


def test_strength_zero_is_identity():
    vol = torch.rand((10, 10, 10), dtype=torch.float64) * 256.0
    sigma = _sigma(vol.shape)
    out = regularize(vol, 3, 1, sigma, factors=(1, 1, 1),
                     mean_preserve=False, strength=0.0)
    assert torch.allclose(out, vol)


def test_strength_increases_smoothing():
    """Larger strength should not decrease the amount the volume is moved
    from its input, on average, for a noisy volume."""
    torch.manual_seed(1)
    vol = torch.rand((14, 14, 14), dtype=torch.float64) * 256.0
    sigma = _sigma(vol.shape, value=3.0)

    out_weak = regularize(vol, 3, 1, sigma, factors=(1, 1, 1),
                          mean_preserve=False, strength=0.5)
    out_strong = regularize(vol, 3, 1, sigma, factors=(1, 1, 1),
                            mean_preserve=False, strength=2.0)

    delta_weak = (out_weak - vol).abs().mean()
    delta_strong = (out_strong - vol).abs().mean()
    assert delta_strong >= delta_weak


def test_floor_mode_scale_is_affine_equivariant():
    """regularize(a*v+b, ..., floor_mode='scale') == a*regularize(v, ...)+b,
    since sigma is scaled by the same factor `a` as the volume."""
    torch.manual_seed(2)
    vol = torch.rand((12, 12, 12), dtype=torch.float64) * 50.0 + 10.0
    sigma = _sigma(vol.shape, value=4.0)

    a, b = 3.0, 7.0

    out = regularize(vol, 2, 1, sigma, factors=(1, 1, 1),
                     mean_preserve=False, floor_mode="scale")
    out_affine = regularize(a * vol + b, 2, 1, a * sigma, factors=(1, 1, 1),
                            mean_preserve=False, floor_mode="scale")

    assert torch.allclose(a * out + b, out_affine, atol=1e-8, rtol=1e-6)


def test_floor_mode_rejects_unknown_value():
    import pytest
    vol = torch.rand((8, 8, 8), dtype=torch.float64)
    sigma = _sigma(vol.shape)
    with pytest.raises(ValueError, match="floor_mode"):
        regularize(vol, 2, 1, sigma, factors=(1, 1, 1), floor_mode="bogus")
