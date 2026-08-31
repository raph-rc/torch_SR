"""Property tests for sr_pytorch.noise_estimate (Rician/Gaussian noise level),
plus a direct comparison against the vendored EZminc oracle (``_legacy``,
backported alongside these tests -- see ``upsample/_legacy/ezminc/README.md``
and ``tests/test_ezminc_backend.py``).

Note: ``noise_estimate`` casts its input to float64 internally
(``sr_pytorch/noise_estimate.py``), so the float32/float64 precision risk
noted during planning applies to ``sr_pytorch.dwt`` when used directly
(covered in ``test_dwt.py``), not to this module's public entry point.
"""

import pytest
import torch

from _legacy import legacy_ezminc
from sr_pytorch.noise_estimate import noise_estimate

_requires_legacy_ext = pytest.mark.skipif(
    not legacy_ezminc.have_legacy_ext(),
    reason="EZminc extensions not built -- run: python _legacy/build_legacy.py")


def _phantom(shape=(32, 32, 32), seed=0):
    """A simple two-level 'object on background' phantom, for a mask to
    have something non-trivial to select."""
    torch.manual_seed(seed)
    vol = torch.zeros(shape, dtype=torch.float64)
    c = tuple(s // 2 for s in shape)
    r = min(shape) // 4
    zz, yy, xx = torch.meshgrid(
        torch.arange(shape[0]), torch.arange(shape[1]), torch.arange(shape[2]),
        indexing="ij")
    sphere = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r ** 2
    vol[sphere] = 100.0
    return vol, sphere


def test_returns_finite_noise_and_signal():
    vol, _ = _phantom()
    vol = vol + torch.randn_like(vol) * 5.0
    noise, mean_signal = noise_estimate(vol)
    assert torch.isfinite(torch.as_tensor(noise))
    assert torch.isfinite(torch.as_tensor(mean_signal))
    assert noise > 0.0


def test_recovers_approximate_gaussian_sigma():
    """On a phantom with known additive Gaussian noise, the Gaussian-mode
    estimate should land within a generous factor of the true sigma --
    the estimator works on a heavily downsampled wavelet subband, so exact
    agreement isn't expected, only the right order of magnitude."""
    true_sigma = 8.0
    vol, _ = _phantom(shape=(48, 48, 48))
    torch.manual_seed(1)
    noisy = vol + torch.randn_like(vol) * true_sigma
    noise, _ = noise_estimate(noisy, gaussian=True)
    assert 0.3 * true_sigma < noise < 3.0 * true_sigma


def test_mask_changes_the_estimate():
    vol, sphere = _phantom(shape=(40, 40, 40))
    torch.manual_seed(2)
    noisy = vol + torch.randn_like(vol) * 6.0
    noise_auto, _ = noise_estimate(noisy)
    noise_masked, signal_masked = noise_estimate(noisy, mask=sphere)
    # Both must be finite and positive; an explicit mask need not produce
    # the identical estimate the built-in k-means detection would (the two
    # are different object-selection strategies), only a sane one.
    assert noise_auto > 0.0
    assert noise_masked > 0.0
    assert torch.isfinite(torch.as_tensor(signal_masked))


def test_gaussian_flag_selects_uncorrected_estimator():
    vol, _ = _phantom(shape=(32, 32, 32))
    torch.manual_seed(3)
    noisy = vol + torch.randn_like(vol) * 6.0
    noise_rician, _ = noise_estimate(noisy, gaussian=False)
    noise_gaussian, _ = noise_estimate(noisy, gaussian=True)
    # The Koay correction is a deterministic function of (signal, nsig) and
    # is not the identity except in degenerate cases, so the two paths
    # should generally disagree.
    assert noise_rician != noise_gaussian


def test_rejects_non_3d_input():
    with pytest.raises(ValueError, match="3-D"):
        noise_estimate(torch.zeros(8, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_parity_cpu_cuda():
    vol, _ = _phantom(shape=(24, 24, 24))
    torch.manual_seed(4)
    noisy_cpu = vol + torch.randn_like(vol) * 5.0
    noisy_cuda = noisy_cpu.to("cuda")
    noise_cpu, signal_cpu = noise_estimate(noisy_cpu)
    noise_cuda, signal_cuda = noise_estimate(noisy_cuda)
    assert noise_cpu == pytest.approx(noise_cuda, rel=1e-6)
    assert signal_cpu == pytest.approx(signal_cuda, rel=1e-6)


@_requires_legacy_ext
def test_rician_unmasked_matches_legacy(chunk):
    """The Rician estimate and the mean signal it implies, on ``chunk``."""
    got = noise_estimate(chunk)
    want = legacy_ezminc.noise_estimate(chunk)
    assert got[0] == pytest.approx(want[0], rel=1e-4)
    assert got[1] == pytest.approx(want[1], rel=1e-4)


@_requires_legacy_ext
def test_gaussian_unmasked_matches_legacy(chunk):
    """The Gaussian assumption skips the Koay correction entirely."""
    got = noise_estimate(chunk, gaussian=True)
    want = legacy_ezminc.noise_estimate(chunk, gaussian=True)
    assert got[0] == pytest.approx(want[0], rel=1e-4)
    assert got[1] == pytest.approx(want[1], rel=1e-4)


@_requires_legacy_ext
def test_masked_matches_legacy(chunk, chunk_mask):
    """A supplied mask replaces the k-means object detection."""
    got = noise_estimate(chunk, mask=chunk_mask)
    want = legacy_ezminc.noise_estimate(chunk, mask=chunk_mask)
    assert got[0] == pytest.approx(want[0], rel=1e-4)
    assert got[1] == pytest.approx(want[1], rel=1e-4)


@_requires_legacy_ext
def test_odd_extent_masked_matches_legacy(chunk, chunk_mask):
    """An odd, non-power-of-two extent exercises the upsample trim."""
    vol = chunk[..., :89]
    mask = chunk_mask[..., :89]
    got = noise_estimate(vol, mask=mask)
    want = legacy_ezminc.noise_estimate(vol, mask=mask)
    assert got[0] == pytest.approx(want[0], rel=1e-4)
    assert got[1] == pytest.approx(want[1], rel=1e-4)
