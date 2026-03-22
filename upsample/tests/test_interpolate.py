"""Tests for sr.interpolate.initial_interpolation."""

import numpy as np
import pytest

from sr.interpolate import initial_interpolation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_error(lr_vol, hr_vol, factors):
    """Return max absolute error of the mean-preservation constraint."""
    fz, fy, fx = factors
    errors = []
    for iz in range(0, hr_vol.shape[0], fz):
        for iy in range(0, hr_vol.shape[1], fy):
            for ix in range(0, hr_vol.shape[2], fx):
                block_mean = hr_vol[iz:iz+fz, iy:iy+fy, ix:ix+fx].mean()
                lr_val = lr_vol[iz//fz, iy//fy, ix//fx]
                errors.append(abs(block_mean - lr_val))
    return max(errors)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factors", [(2, 2, 2), (3, 3, 3), (2, 3, 2)])
def test_output_shape(factors):
    vol = np.random.default_rng(0).random((10, 12, 14))
    result = initial_interpolation(vol, factors)
    expected = tuple(s * f for s, f in zip(vol.shape, factors))
    assert result.shape == expected, f"Got {result.shape}, expected {expected}"


def test_output_shape_asymmetric():
    vol = np.random.default_rng(1).random((8, 16, 24))
    result = initial_interpolation(vol, (2, 2, 2))
    assert result.shape == (16, 32, 48)


# ---------------------------------------------------------------------------
# Mean-preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factors", [(2, 2, 2), (3, 3, 3)])
def test_mean_preservation(factors):
    rng = np.random.default_rng(42)
    vol = rng.random((8, 8, 8)) * 100.0
    hr = initial_interpolation(vol, factors)
    max_err = _mean_error(vol, hr, factors)
    # Numerical tolerance: mean correction is exact up to floating-point noise.
    assert max_err < 1e-10, f"Mean preservation error: {max_err}"


# ---------------------------------------------------------------------------
# No NaN / Inf
# ---------------------------------------------------------------------------

def test_no_nan_inf():
    vol = np.random.default_rng(7).random((10, 10, 10))
    result = initial_interpolation(vol, (2, 2, 2))
    assert np.all(np.isfinite(result)), "Result contains NaN or Inf"


# ---------------------------------------------------------------------------
# Uniform volume — result should be (nearly) uniform
# ---------------------------------------------------------------------------

def test_uniform_volume():
    vol = np.ones((6, 6, 6)) * 5.0
    result = initial_interpolation(vol, (2, 2, 2))
    assert np.allclose(result, 5.0, atol=1e-10), "Uniform volume not preserved"


# ---------------------------------------------------------------------------
# Factor=1 is identity (no upsampling)
# ---------------------------------------------------------------------------

def test_factor_one_identity():
    rng = np.random.default_rng(3)
    vol = rng.random((8, 8, 8))
    result = initial_interpolation(vol, (1, 1, 1))
    assert result.shape == vol.shape
    assert np.allclose(result, vol, atol=1e-10)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_invalid_ndim():
    with pytest.raises(ValueError, match="3-D"):
        initial_interpolation(np.ones((4, 4)), (2, 2, 2))


def test_invalid_factor():
    with pytest.raises(ValueError, match="Factor"):
        initial_interpolation(np.ones((4, 4, 4)), (2, 0, 2))
