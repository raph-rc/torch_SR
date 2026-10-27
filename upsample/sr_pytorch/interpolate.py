"""
Initial spline interpolation step for NLM upsampling.

Implements ``InitialInterpolation()`` from NLMUpsample2.m (Manjon 2010):

1. Cubic spline interpolation with LR samples placed at the *centres* of
   their corresponding HR blocks (not at the HR grid origin).
2. Boundary replication: edge slices are replicated to fill the half-block
   margin that lies outside the extrapolation range.
3. Mean correction: for every HR block the mean is shifted to match the
   corresponding LR voxel value, preserving image statistics.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def initial_interpolation(
    lr_vol: np.ndarray,
    factors: Tuple[int, int, int],
) -> np.ndarray:
    """Cubic spline upsample with block-centre LR placement and mean correction.

    The LR voxel at index ``(iz, iy, ix)`` is placed at HR coordinate
    ``(iz*fz + (fz-1)/2,  iy*fy + (fy-1)/2,  ix*fx + (fx-1)/2)``.
    This matches Matlab's ``ori = (1+lf)/2`` convention (converted to 0-based).

    Args:
        lr_vol: Low-resolution volume, shape (D, H, W), float64.
        factors: Integer upsampling factors (fz, fy, fx).

    Returns:
        High-resolution volume, shape (D*fz, H*fy, W*fx).

    Raises:
        ValueError: If *lr_vol* is not 3-D or any factor is < 1.
    """
    if lr_vol.ndim != 3:
        raise ValueError(
            f"Expected 3-D volume, got shape {tuple(lr_vol.shape)}"
        )
    fz, fy, fx = int(factors[0]), int(factors[1]), int(factors[2])
    for name, val in (("fz", fz), ("fy", fy), ("fx", fx)):
        if val < 1:
            raise ValueError(f"Factor {name} must be >= 1, got {val}")

    vol = np.asarray(lr_vol, dtype=np.float64)
    shape_lr = np.array(vol.shape)            # (Dz, Dy, Dx)
    f_arr = np.array([fz, fy, fx], dtype=float)
    shape_hr = (shape_lr * f_arr).astype(int)  # (Dz*fz, ...)

    # LR sample positions in HR coordinate system (0-based).
    # LR index i  →  HR position  i*f + (f-1)/2
    coords_lr = [
        np.arange(n) * fi + (fi - 1.0) / 2.0
        for n, fi in zip(shape_lr, [fz, fy, fx])
    ]

    interp = RegularGridInterpolator(
        coords_lr,
        vol,
        method="cubic",
        bounds_error=False,
        fill_value=None,   # extrapolate beyond boundary
    )

    # Evaluate at every HR voxel position.
    grids = [np.arange(n) for n in shape_hr]
    mg = np.meshgrid(*grids, indexing="ij")
    pts = np.stack([g.ravel() for g in mg], axis=1)  # (N, 3)
    hr_vol = interp(pts).reshape(shape_hr)

    # ---- Boundary correction -----------------------------------------------
    # The first/last floor(f/2) slices along each axis are outside the LR
    # sample range; replicate the nearest valid slice (matches Matlab).
    hr_vol = hr_vol.copy()
    for d, half in enumerate([fz // 2, fy // 2, fx // 2]):
        n = shape_hr[d]
        if half == 0:
            continue
        # Build generic slice helpers for dimension d.
        def _slc(idx):
            s = [slice(None)] * 3
            s[d] = idx
            return tuple(s)

        src_lead = hr_vol[_slc(slice(half, half + 1))]
        hr_vol[_slc(slice(0, half))] = src_lead

        src_tail = hr_vol[_slc(slice(n - half - 1, n - half))]
        hr_vol[_slc(slice(n - half, n))] = src_tail

    # ---- Mean correction ---------------------------------------------------
    # For every HR block of shape (fz, fy, fx), add an offset so that the
    # block mean equals the value of the corresponding LR voxel.
    result = hr_vol.copy()
    for iz in range(0, shape_hr[0], fz):
        for iy in range(0, shape_hr[1], fy):
            for ix in range(0, shape_hr[2], fx):
                lr_val = vol[iz // fz, iy // fy, ix // fx]
                block = hr_vol[iz : iz + fz, iy : iy + fy, ix : ix + fx]
                result[iz : iz + fz, iy : iy + fy, ix : ix + fx] += (
                    lr_val - block.mean()
                )

    return result
