"""Integration tests for sr.core.nlm_upsample (requires C extension)."""

import numpy as np
import pytest

from sr.regularize import have_c_ext

# Skip when the C extension is absent.
pytestmark = pytest.mark.skipif(
    not have_c_ext(),
    reason="C extension not built — run: python sr/_nlm_build.py",
)

from sr.core import nlm_upsample  # noqa: E402 (import after skip-check)


# ---------------------------------------------------------------------------
# Basic integration: shape and finite values
# ---------------------------------------------------------------------------

def test_output_shape_isotropic():
    rng = np.random.default_rng(42)
    lr = rng.random((10, 10, 10)) * 100.0
    hr = nlm_upsample(lr, factors=(2, 2, 2))
    assert hr.shape == (20, 20, 20)
    assert np.all(np.isfinite(hr))


def test_output_shape_anisotropic():
    rng = np.random.default_rng(43)
    lr = rng.random((8, 12, 16)) * 200.0
    hr = nlm_upsample(lr, factors=(2, 2, 3))
    assert hr.shape == (16, 24, 48)


# ---------------------------------------------------------------------------
# Intensity range: result should be in the same ballpark as input
# ---------------------------------------------------------------------------

def test_intensity_range():
    rng = np.random.default_rng(7)
    lr = rng.random((10, 10, 10)) * 100.0
    hr = nlm_upsample(lr, factors=(2, 2, 2))
    # Spline interpolation and NLM can overshoot slightly beyond input range.
    # We just verify the result is finite and stays within a loose bound.
    assert np.all(np.isfinite(hr)), "Result contains NaN or Inf"
    assert hr.min() >= -0.6 * lr.max(), "Large negative overshoot"
    assert hr.max() <= 2.0 * lr.max(), "Large positive overshoot"


# ---------------------------------------------------------------------------
# Uniform input
# ---------------------------------------------------------------------------

def test_uniform_input():
    lr = np.ones((8, 8, 8)) * 50.0
    hr = nlm_upsample(lr, factors=(2, 2, 2))
    assert np.allclose(hr, 50.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_wrong_ndim():
    with pytest.raises(ValueError, match="3-D"):
        nlm_upsample(np.ones((4, 4)))


def test_zero_volume():
    with pytest.raises(ValueError, match="all-zero"):
        nlm_upsample(np.zeros((4, 4, 4)))
