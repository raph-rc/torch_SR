"""Stage 1: the vendored EZminc denoisers really are the EZminc algorithms.

Compares the CFFI shim in ``_legacy.legacy_ezminc`` against the *installed*
``mincnlm``, ``minc_anlm`` and ``noise_estimate`` programs, so that
``legacy_ezminc``'s entry points can be trusted as an in-process oracle.
Ported from ``torch_n3/tests/test_ezminc_backend.py``.

Everything was recorded at one thread: ``mincnlm``'s default of four is
nondeterministic, and ``minc_anlm``'s own help warns that its multithreaded
output differs. See ``tests/regenerate_reference.py``.

**Not tested here, deliberately**: whether ``mincnlm``/``minc_anlm`` (Coupe's
non-local-means convention) numerically match ``sr_pytorch.regularize``/
``sr/nlm_core.c`` (Manjon's convention, ``cMRegularizarNLM3D_V2.c``). They do
not, and are not meant to -- see ``COPYING.NLM`` and
``upsample/_legacy/ezminc/README.md``. This file validates the shim against
its own oracle, not this project's own NLM filter against a different one.
"""

import pytest
import torch

from _legacy import legacy_ezminc
from tests.conftest import relative_rms

pytestmark = pytest.mark.skipif(
    not legacy_ezminc.have_legacy_ext(),
    reason="EZminc extensions not built -- run: python _legacy/build_legacy.py")


def test_noise_estimate_matches_installed(legacy_output, chunk):
    """The Rician-mode noise and the mean signal it implies, on ``chunk``."""
    noise, mean_signal = legacy_ezminc.noise_estimate(chunk, gaussian=False)

    # The program prints six significant digits, so a 1e-4 bound is agreement
    # to its last printed digit.
    assert noise == pytest.approx(
        float(legacy_output.scalar("noise_estimate.chunk_noise")), rel=1e-4)
    assert mean_signal / noise == pytest.approx(
        float(legacy_output.scalar("noise_estimate.chunk_snr")), rel=1e-4)


def test_noise_estimate_gaussian(legacy_output, chunk):
    """The Gaussian assumption skips the Koay correction entirely."""
    noise, mean_signal = legacy_ezminc.noise_estimate(chunk, gaussian=True)

    assert noise == pytest.approx(
        float(legacy_output.scalar("noise_estimate.chunk_noise_gauss")),
        rel=1e-4)
    assert mean_signal / noise == pytest.approx(
        float(legacy_output.scalar("noise_estimate.chunk_snr_gauss")),
        rel=1e-4)


def test_noise_estimate_masked(legacy_output, chunk, chunk_mask):
    """A supplied mask replaces the k-means object detection."""
    noise, mean_signal = legacy_ezminc.noise_estimate(chunk, gaussian=False,
                                                       mask=chunk_mask)

    assert noise == pytest.approx(
        float(legacy_output.scalar("noise_estimate.chunk_mask_noise")),
        rel=1e-4)
    assert mean_signal / noise == pytest.approx(
        float(legacy_output.scalar("noise_estimate.chunk_mask_snr")),
        rel=1e-4)


def test_mincnlm_with_explicit_sigma(legacy_output, chunk):
    """``mincnlm -mt 1 -sigma 20000`` on the same input, through the shim.

    The reference came out of a 16-bit MINC file, so the bound is set by that
    quantisation, not by the arithmetic.
    """
    denoised = legacy_ezminc.mincnlm(chunk, sigma=20000.0)
    assert relative_rms(denoised, legacy_output["mincnlm.chunk_sigma"]) < 1e-4


def test_mincnlm_automatic_sigma(legacy_output, chunk):
    """Without a sigma the program estimates one (Gaussian) noise level."""
    denoised = legacy_ezminc.mincnlm(chunk, sigma=0.0)
    assert relative_rms(denoised, legacy_output["mincnlm.chunk_auto"]) < 1e-4


def test_minc_anlm_matches_installed(legacy_output, chunk):
    """``minc_anlm --search 2 --patch 1 --float``, through the shim."""
    fima = legacy_ezminc.minc_anlm(chunk)
    assert relative_rms(fima, legacy_output["minc_anlm.chunk"]) < 1e-5


def test_mincnlm_is_deterministic_at_one_thread(chunk):
    """Two calls with the same arguments produce identical bytes."""
    first = legacy_ezminc.mincnlm(chunk, sigma=20000.0)
    second = legacy_ezminc.mincnlm(chunk, sigma=20000.0)
    assert torch.equal(first, second)


def test_mincnlm_rejects_speckle_without_sigma(chunk):
    """``-w 1`` needs an explicit sigma; Exec would leave the output blank."""
    with pytest.raises(ValueError):
        legacy_ezminc.mincnlm(chunk, weight_method=1, sigma=0.0)


def test_mincnlm_rejects_block_without_patch_cover(chunk):
    """``block`` requires ``S >= b_space/2``; Exec would ``exit(0)``."""
    with pytest.raises(ValueError):
        legacy_ezminc.mincnlm(chunk, block=1, S=1, b_space=4)


def test_mincnlm_rejects_anisotropic_block(chunk):
    """The block version cannot run in anisotropic mode."""
    with pytest.raises(ValueError):
        legacy_ezminc.mincnlm(chunk, block=1, anisotropic=True)
