"""
Optimization-based NLM super-resolution (PyTorch).  **EXPERIMENTAL.**

This method is slower than iterative NLM and does not outperform it in PSNR.
It is kept for research purposes only and is not exposed by the default CLI.

Instead of iterating the NLM regularization step to convergence, this module
optimizes an additive correction field δ in HR space via Adam to minimize the
data-fidelity loss:

    L(δ) = MSE( downsample( regularize( HR_init + δ ) ), LR )

where ``downsample`` is ``F.avg_pool3d`` (differentiable block-averaging) and
``regularize`` uses ``detach_weights=True`` so gradients flow through the
weighted-sum values without retaining the weight-computation graph.

Usage
-----
    from sr_pytorch.optimize import nlm_sr_optimize
    hr = nlm_sr_optimize(lr_vol, factors=(2, 1, 1), n_steps=100, verbose=True)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from sr.interpolate import initial_interpolation
from .regularize import regularize
from .core import _noise_level

logger = logging.getLogger(__name__)


def nlm_sr_optimize(
    lr_vol: np.ndarray,
    factors: Tuple[int, int, int] = (2, 2, 2),
    v: int = 3,
    f: int = 1,
    n_steps: int = 100,
    lr_rate: float = 0.02,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Optimization-based NLM super-resolution.

    Learns an additive correction field δ on top of the spline initial estimate
    by minimizing ``MSE(downsample(regularize(HR_init + δ)), LR)`` with Adam.

    Args:
        lr_vol:   Low-resolution input volume, shape ``(D, H, W)``.
        factors:  Integer upsampling factors ``(fz, fy, fx)``.
        v:        NLM search-neighbourhood radius (default 3).
        f:        NLM patch half-size (default 1 → 3×3×3 patch).
        n_steps:  Number of Adam optimization steps (default 100).
        lr_rate:  Adam learning rate (default 0.5).
        device:   PyTorch device.  Defaults to CUDA if available, else CPU.
        dtype:    Floating-point dtype.  Defaults to ``float32`` on CUDA,
                  ``float64`` on CPU.
        verbose:  Print per-step loss if ``True``.

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
        "NLM optimize (PyTorch/%s %s): input %s → factors %s",
        device, dtype, tuple(vol_np.shape), factors,
    )

    # ---- initial interpolation (scipy cubic, on CPU) ----------------------
    hr_init_np = initial_interpolation(vol_np, factors)
    logger.info("Initial interpolation done, HR shape %s", tuple(hr_init_np.shape))

    # ---- move to device ---------------------------------------------------
    hr_init_t = torch.from_numpy(hr_init_np).to(device=device, dtype=dtype)

    # ---- noise level (fixed throughout optimization) ----------------------
    with torch.no_grad():
        sigma = _noise_level(hr_init_t)

    # ---- loss target: actual normalised LR volume
    # (NOT avg_pool3d(HR_init) — that would be trivially zero at δ=0 because
    #  the spline initialisation already satisfies avg_pool3d(HR_init) = LR_norm
    #  and mean_preserve=False below breaks that equality inside regularize)
    fz, fy, fx = factors
    lr_target = torch.from_numpy(vol_np).to(device=device, dtype=dtype)

    # ---- additive correction field ----------------------------------------
    delta = torch.zeros_like(hr_init_t, requires_grad=True)
    optimizer = optim.Adam([delta], lr=lr_rate)

    logger.info(
        "Optimizing δ: n_steps=%d  lr=%.4f  device=%s  dtype=%s",
        n_steps, lr_rate, device, dtype,
    )

    for step in range(n_steps):
        hr_cand = hr_init_t + delta

        # mean_preserve=False so the forward model is non-trivial:
        # block-average of NLM-smoothed output ≠ block-average of input,
        # giving the optimizer real gradient signal.
        hr_reg = regularize(
            hr_cand, v, f, sigma, factors,
            mean_preserve=False,
            detach_weights=True,
            no_grad=False,
        )

        lr_pred = F.avg_pool3d(
            hr_reg.unsqueeze(0).unsqueeze(0),
            kernel_size=(fz, fy, fx),
            stride=(fz, fy, fx),
        ).squeeze(0).squeeze(0)

        loss = F.mse_loss(lr_pred, lr_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        logger.info("Step %d/%d: loss=%.6f", step + 1, n_steps, loss_val)
        if verbose:
            print(f"  step {step + 1:4d}/{n_steps}: loss={loss_val:.6f}")

    # ---- rescale and return as float64 numpy ------------------------------
    with torch.no_grad():
        output = (hr_init_t + delta) * (m / 256.0)
    return output.cpu().to(torch.float64).numpy()
