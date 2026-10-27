"""
NLM super-resolution upsampling – PyTorch pipeline.

Drop-in replacement for ``sr.core.nlm_upsample`` that runs entirely in
PyTorch (no compiled C extension) and is GPU-capable.

Differences from the C-backed implementation
--------------------------------------------
* The regularization kernel is vectorised over search offsets using
  ``avg_pool3d`` instead of a per-voxel C loop.
* Defaults to ``torch.float32`` on CUDA (≈ 2× faster than float64 on most
  GPUs with negligible PSNR impact) and ``torch.float64`` on CPU to match
  the C reference.
* The noise-level and local-std helpers use ``avg_pool3d`` with reflect
  padding, matching ``scipy.ndimage.uniform_filter(mode='reflect')``.
* Initial interpolation reuses the same scipy cubic B-spline code as the
  C pipeline (``sr.interpolate.initial_interpolation``); only the NLM
  regularization loop is replaced.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# from sr.interpolate import initial_interpolation
from interpolate import initial_interpolation
# from .regularize import regularize
from regularize import regularize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noise-level helpers (PyTorch equivalents of sr.core helpers)
# ---------------------------------------------------------------------------

def _local_std(vol: torch.Tensor) -> torch.Tensor:
    """Population std in a 3×3×3 neighbourhood (reflect BC)."""
    vol5 = vol.unsqueeze(0)
    padded    = F.pad(vol5,        [1] * 6, mode="reflect")
    padded_sq = F.pad(vol5.pow(2), [1] * 6, mode="reflect")
    lmean    = F.avg_pool3d(padded,    kernel_size=3, stride=1, padding=0)
    lsqmean  = F.avg_pool3d(padded_sq, kernel_size=3, stride=1, padding=0)
    return (lsqmean - lmean.pow(2)).clamp(min=0.0).sqrt().squeeze(0)


def _noise_level(vol: torch.Tensor) -> torch.Tensor:
    """Per-voxel noise level: smooth local std then halve."""
    sigma = _local_std(vol)
    sig4 = sigma.unsqueeze(0)
    padded = F.pad(sig4, [1] * 6, mode="reflect")
    smoothed = F.avg_pool3d(padded, kernel_size=3, stride=1, padding=0)
    return (smoothed / 2.0).squeeze(0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def nlm_upsample(
    lr_vol: np.ndarray,
    factors: Tuple[int, int, int] = (2, 2, 2),
    v: int = 3,
    f: int = 1,
    tol: float = 1.2,
    max_iter: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> np.ndarray:
    """Upsample a 3-D MRI volume using iterative PyTorch NLM regularization.

    Replicates the full pipeline of ``sr.core.nlm_upsample`` without any
    compiled C extension.

    Args:
        lr_vol:   Low-resolution input, shape ``(D, H, W)``.
        factors:  Integer upsampling factors ``(fz, fy, fx)``.
        v:        Search-neighbourhood radius (default 3).
        f:        Patch half-size (default 1 → 3×3×3 patch).
        tol:      Convergence ratio threshold (default 1.2).
        max_iter: Stop after *N* iterations (0 = run to convergence).
        device:   PyTorch device.  Defaults to CUDA if available, else CPU.
        dtype:    Floating-point dtype.  Defaults to ``float32`` on CUDA,
                  ``float64`` on CPU (matching the C reference).

    Returns:
        Upsampled volume, shape ``(D*fz, H*fy, W*fx)``, float64 numpy array.
    """
    if lr_vol.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {tuple(lr_vol.shape)}")

    # ---- resolve device / dtype -------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if dtype is None:
        dtype = torch.float32 if device.type == "cuda" else torch.float64

    # ---- normalise to [0, 256] --------------------------------------------
    m = float(np.max(lr_vol))
    if m == 0.0:
        raise ValueError("Input volume has all-zero intensities.")
    vol_np = lr_vol.astype(np.float64) * (256.0 / m)

    logger.info(
        "NLM upsample (PyTorch/%s %s): input %s → factors %s",
        device, dtype, tuple(vol_np.shape), factors,
    )

    # ---- initial interpolation (scipy cubic, on CPU) ----------------------
    F_np = initial_interpolation(vol_np, factors)
    logger.info("Initial interpolation done, HR shape %s", tuple(F_np.shape))

    # ---- move to device ---------------------------------------------------
    Ft = torch.from_numpy(F_np).to(device=device, dtype=dtype)

    # ---- noise level estimate ---------------------------------------------
    level = _noise_level(Ft)

    last    = Ft.clone()
    d_hist:  list[float] = []
    ds_hist: list[float] = []
    down = 0

    logger.info(
        "Iterating NLM regularization (v=%d, f=%d, tol=%.2f, device=%s)",
        v, f, tol, device,
    )

    ii   = 0
    lima = Ft
    while True:
        lima = regularize(Ft, v, f, level, factors)

        di = float((Ft - lima).abs().mean())
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
                ds_val = float((last - lima).abs().mean())
                ds_hist.append(ds_val)
                if len(ds_hist) > 1:
                    ds_ratio = ds_hist[-2] / (ds_hist[-1] + 1e-12)
                    if ds_ratio < tol:
                        logger.info(
                            "Converged (ds criterion) after %d iterations", ii + 1
                        )
                        break
                last = lima.clone()
            else:
                down = 0

            if di <= 0.001:
                logger.info("Converged (d<=0.001) after %d iterations", ii + 1)
                break

        Ft = lima
        ii += 1

    # ---- rescale and return as float64 numpy ------------------------------
    return (lima * (m / 256.0)).cpu().to(torch.float64).numpy()
