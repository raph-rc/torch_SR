"""Tests for sr.regularize (requires compiled C extension)."""

import numpy as np
import pytest

from sr.regularize import have_c_ext, regularize

# Skip the entire module if the C extension has not been compiled.
pytestmark = pytest.mark.skipif(
    not have_c_ext(),
    reason="C extension not built — run: python sr/_nlm_build.py",
)


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(20, 20, 20), (16, 24, 32)])
def test_output_shape(shape):
    rng = np.random.default_rng(0)
    vol = rng.random(shape) * 256.0
    sigma = np.ones(shape) * 5.0
    result = regularize(vol, v=3, f=1, sigma=sigma, factors=(2, 2, 2))
    assert result.shape == shape


# ---------------------------------------------------------------------------
# No NaN / Inf with realistic sigma
# ---------------------------------------------------------------------------

def test_no_nan_inf():
    rng = np.random.default_rng(1)
    vol = rng.random((16, 16, 16)) * 256.0
    sigma = np.full(vol.shape, 10.0)
    result = regularize(vol, v=2, f=1, sigma=sigma, factors=(2, 2, 2))
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# With sigma < 1 everywhere the kernel skips all voxels and returns ima
# ---------------------------------------------------------------------------

def test_sigma_below_threshold_returns_identity():
    rng = np.random.default_rng(2)
    vol = rng.random((10, 10, 10)) * 256.0
    # sigma < 1 → ThreadFunc skips all voxels; fima is initialised to ima.
    sigma = np.full(vol.shape, 0.5)
    result = regularize(vol, v=3, f=1, sigma=sigma, factors=(2, 2, 2))
    assert np.allclose(result, vol)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_wrong_volume_ndim():
    with pytest.raises(ValueError, match="3-D"):
        regularize(np.ones((4, 4)), v=1, f=1,
                   sigma=np.ones((4, 4)), factors=(2, 2, 2))


def test_sigma_shape_mismatch():
    with pytest.raises(ValueError, match="sigma shape"):
        regularize(np.ones((8, 8, 8)), v=1, f=1,
                   sigma=np.ones((4, 4, 4)), factors=(2, 2, 2))
