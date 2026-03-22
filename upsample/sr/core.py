"""
NLM super-resolution upsampling pipeline.

Re-implements ``NLMUpsample2.m`` (Manjon & Coupe 2010) in Python,
calling the compiled C kernel for the inner regularization loop.

Reference
---------
Manjon JV, Coupe P, Buades A, Collins DL, Robles M.
MRI superresolution using self-similarity and image priors.
*Int J Biomed Imaging*, 2010.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import scipy.ndimage

from sr.interpolate import initial_interpolation
from sr.regularize import regularize

logger = logging.getLogger(__name__)


def _local_std(vol: np.ndarray) -> np.ndarray:
    """Population standard deviation in a 3×3×3 neighbourhood.

    Equivalent to Matlab's ``stdfilt(vol, ones(3,3,3))``.

    Args:
        vol: 3-D array.

    Returns:
        Array of the same shape with per-voxel local std.
    """
    # Var[X] = E[X²] - (E[X])²
    mode = "reflect"
    local_mean = scipy.ndimage.uniform_filter(vol, size=3, mode=mode)
    local_sq_mean = scipy.ndimage.uniform_filter(vol ** 2, size=3, mode=mode)
    return np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0.0))


def _noise_level(vol: np.ndarray) -> np.ndarray:
    """Estimate per-voxel noise level as in NLMUpsample2.m.

    Matches::

        sigma = stdfilt(bima, ones(3,3,3))
        pad   = padarray(sigma, [1,1,1], 'symmetric')
        sigma = convn(pad, ones(3,3,3), 'valid') / 27
        level = sigma / 2

    The second smoothing pass (convn after padarray) is equivalent to
    ``uniform_filter`` with ``mode='reflect'``.

    Args:
        vol: 3-D volume normalised to [0, 256].

    Returns:
        Per-voxel noise level array, same shape as *vol*.
    """
    sigma = _local_std(vol)
    smoothed = scipy.ndimage.uniform_filter(sigma, size=3, mode="reflect")
    return smoothed / 2.0


def nlm_upsample(
    lr_vol: np.ndarray,
    factors: Tuple[int, int, int] = (2, 2, 2),
    v: int = 3,
    f: int = 1,
    tol: float = 1.2,
    max_iter: int = 0,
) -> np.ndarray:
    """Upsample a 3-D MRI volume using iterative NLM regularization.

    This is the main entry point; it reproduces the full pipeline of
    ``NLMUpsample2.m``:

    1. Normalise intensities to [0, 256].
    2. Initial cubic-spline interpolation with mean correction
       (``InitialInterpolation``).
    3. Estimate per-voxel noise level from the initial estimate.
    4. Iterate NLM regularization (``cMRegularizarNLM3D_V2``), halving the
       noise level when convergence slows, until two successive convergence
       criteria are met.
    5. Rescale back to original intensity range.

    Args:
        lr_vol: Low-resolution input, shape (D, H, W).
        factors: Integer upsampling factors ``(fz, fy, fx)`` matching the
            ``(D, H, W)`` axis order.
        v: Search-neighbourhood radius (Matlab default: 3).
        f: Patch half-size; patch covers ``(2f+1)^3`` voxels (Matlab
            default: 1 → 3×3×3 patch).
        tol: Convergence ratio threshold.  Iteration continues until the
            ratio of successive mean absolute differences drops below *tol*
            twice in a multi-scale sense (Matlab default: 1.2).

    Returns:
        Upsampled volume, shape ``(D*fz, H*fy, W*fx)``, same dtype as
        *lr_vol* after float64 round-trip.

    Raises:
        ValueError: If *lr_vol* is not 3-D or has all-zero intensities.
        ImportError: If the C extension has not been compiled.
    """
    if lr_vol.ndim != 3:
        raise ValueError(
            f"Expected 3-D volume, got shape {tuple(lr_vol.shape)}"
        )

    # Normalise to [0, 256] — matches Matlab ``ima = ima * 256 / m``.
    m = float(np.max(lr_vol))
    if m == 0.0:
        raise ValueError("Input volume has all-zero intensities.")
    vol = lr_vol.astype(np.float64) * (256.0 / m)

    logger.info(
        "NLM upsample: input shape %s, factors %s", tuple(vol.shape), factors
    )

    # Step 1 — initial interpolation.
    F = initial_interpolation(vol, factors)
    logger.info("Initial interpolation done, HR shape %s", tuple(F.shape))

    # Step 2 — noise level estimate.
    level = _noise_level(F)

    last = F.copy()
    d_hist: list[float] = []
    ds_hist: list[float] = []
    down = 0

    logger.info(
        "Iterating NLM regularization (v=%d, f=%d, tol=%.2f)", v, f, tol
    )

    ii = 0
    lima = F  # will be overwritten on first iteration
    while True:
        lima = regularize(F, v, f, level, factors)

        di = float(np.mean(np.abs(F - lima)))
        d_hist.append(di)
        logger.info("Iteration %d: mean|F-lima|=%.6f", ii + 1, di)

        if max_iter > 0 and (ii + 1) >= max_iter:
            logger.info("Stopping: reached max_iter=%d", max_iter)
            break

        if ii > 0:
            ratio = d_hist[-2] / (d_hist[-1] + 1e-12)
            if ratio < tol and down == 0:
                down = 1
                level = level / 2.0
                ds_val = float(np.mean(np.abs(last - lima)))
                ds_hist.append(ds_val)
                logger.debug(
                    "Level halved; ds=%.6f (ds history len=%d)",
                    ds_val,
                    len(ds_hist),
                )
                if len(ds_hist) > 1:
                    ds_ratio = ds_hist[-2] / (ds_hist[-1] + 1e-12)
                    if ds_ratio < tol:
                        logger.info(
                            "Converged (ds criterion) after %d iterations",
                            ii + 1,
                        )
                        break
                last = lima.copy()
            else:
                down = 0

            if di <= 0.001:
                logger.info(
                    "Converged (d<=0.001) after %d iterations", ii + 1
                )
                break

        F = lima
        ii += 1

    # Rescale back to original range.
    return lima * (m / 256.0)
