"""
Python wrapper around the compiled C NLM regularization kernel.

Requires the C extension to be built first::

    python sr/_nlm_build.py
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from sr._nlm_core import ffi as _ffi
    from sr._nlm_core import lib as _lib

    _HAVE_C_EXT = True
except ImportError:
    _HAVE_C_EXT = False
    _ffi = None  # type: ignore[assignment]
    _lib = None  # type: ignore[assignment]


def have_c_ext() -> bool:
    """Return True if the C extension has been compiled and is importable."""
    return _HAVE_C_EXT


def regularize(
    volume: np.ndarray,
    v: int,
    f: int,
    sigma: np.ndarray,
    factors: Tuple[int, int, int],
) -> np.ndarray:
    """Apply one NLM regularization pass to a 3-D volume.

    Calls the compiled C kernel ``nlm3d_regularize``.  The kernel is the
    direct C equivalent of the Matlab MEX function ``cMRegularizarNLM3D_V2``
    from Manjon (2010).

    Args:
        volume: Input volume, shape (D, H, W), float64.
        v: Search-neighbourhood radius.
        f: Patch half-size; patch covers (2f+1)^3 voxels.
        sigma: Per-voxel noise-level estimate, same shape as *volume*.
        factors: Integer upsampling factors (fz, fy, fx) matching the
            dimension order of *volume*.

    Returns:
        Regularized volume, same shape and dtype as *volume*.

    Raises:
        ImportError: If the C extension has not been compiled.
        ValueError: If *volume* or *sigma* have unexpected shapes.
        RuntimeError: If the C kernel reports a memory-allocation failure.
    """
    if not _HAVE_C_EXT:
        raise ImportError(
            "C extension not built.  Run:  python sr/_nlm_build.py"
        )

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3-D volume, got shape {tuple(volume.shape)}"
        )
    if sigma.shape != volume.shape:
        raise ValueError(
            f"sigma shape {tuple(sigma.shape)} != volume shape {tuple(volume.shape)}"
        )

    # Ensure C-contiguous float64 — copies only if needed.
    vol = np.ascontiguousarray(volume, dtype=np.float64)
    sig = np.ascontiguousarray(sigma, dtype=np.float64)
    out = np.empty(vol.size, dtype=np.float64)

    nz, ny, nx = vol.shape
    fac_z, fac_y, fac_x = int(factors[0]), int(factors[1]), int(factors[2])

    vol_ptr = _ffi.cast("const double *", _ffi.from_buffer(vol))
    sig_ptr = _ffi.cast("const double *", _ffi.from_buffer(sig))
    out_ptr = _ffi.cast("double *", _ffi.from_buffer(out))

    ret = _lib.nlm3d_regularize(
        vol_ptr,
        nx, ny, nz,
        v, f,
        sig_ptr,
        fac_x, fac_y, fac_z,
        out_ptr,
    )

    if ret != 0:
        raise RuntimeError("nlm3d_regularize: internal memory allocation failed")

    return out.reshape(nz, ny, nx)
