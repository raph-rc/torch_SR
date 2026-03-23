"""
PyTorch NLM regularization kernel.

Vectorised reimplementation of ``cMRegularizarNLM3D_V2`` that runs on any
PyTorch device (CPU or CUDA) without a compiled C extension.

Algorithm
---------
For every search offset (dz, dy, dx) in the (2v+1)³ neighbourhood:
  1. Extract the shifted volume from a pre-padded buffer.
  2. Compute per-voxel patch distance with ``avg_pool3d``
     (patch size (2f+1)³, no stride, no extra padding).
  3. Apply the NLM weight  w = exp(-max(d/(2σ²) - 1, 0)).
  4. Optional fast rejection: skip weight if
     |local_mean(p) - local_mean(q)| > 0.6 · σ(p).
  5. Symmetrically accumulate weighted values and weights.
After the loop, normalise by total weight and apply the mean-preservation
correction (each (fz×fy×fx) output block is shifted to match its input
block mean).

Gradient modes
--------------
``detach_weights=True`` (default)
    NLM weights are detached from the autograd graph before accumulation.
    The output is a spatially-varying linear combination of input voxels
    with fixed (non-differentiable) coefficients.  Gradients flow through
    the *values* only, keeping memory at inference level (no graph stored
    across loop iterations).  Use this when the filter sits inside a
    training loop.

``detach_weights=False``
    Full autograd graph is retained through weight computation.  Gradients
    are exact but memory scales as O(iterations) — ~39 GB for v=3,f=1 on a
    200×200×250 volume.  Only practical for very small volumes or when
    exact weight gradients are explicitly required.

``no_grad=True`` (default)
    Wraps the entire call in ``torch.no_grad()``.  No graph is built at
    all; use for inference / upsampling pipelines where gradients are not
    needed.  Setting ``no_grad=False`` while keeping ``detach_weights=True``
    is the recommended mode for use inside a training loop.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def regularize(
    volume: torch.Tensor,
    v: int,
    f: int,
    sigma: torch.Tensor,
    factors: Tuple[int, int, int],
    mean_preserve: bool = True,
    detach_weights: bool = True,
    no_grad: bool = True,
) -> torch.Tensor:
    """NLM regularization – PyTorch implementation.

    Args:
        volume:         Input volume, shape ``(D, H, W)``, float.
        v:              Search-neighbourhood radius.
        f:              Patch half-size; patch covers ``(2f+1)³`` voxels.
        sigma:          Per-voxel noise-level estimate, same shape as *volume*.
        factors:        Upsampling factors ``(fz, fy, fx)``.
        mean_preserve:  When ``True`` (default) each ``(fz×fy×fx)`` output
                        block is shifted to match its input block mean.
                        Set to ``False`` for pure per-voxel NLM filtering.
        detach_weights: When ``True`` (default) NLM weights are detached from
                        the autograd graph; gradients flow through the weighted
                        sum of values only.  Set to ``False`` for full
                        gradient support (high memory cost).
        no_grad:        When ``True`` (default) the entire call runs inside
                        ``torch.no_grad()``, building no graph at all.
                        Set to ``False`` when calling from a training loop.

    Returns:
        Regularized volume, same shape and device as *volume*.
    """
    if no_grad:
        with torch.no_grad():
            return _regularize(volume, v, f, sigma, factors,
                               mean_preserve, detach_weights=False)
    return _regularize(volume, v, f, sigma, factors,
                       mean_preserve, detach_weights)


def _regularize(
    volume: torch.Tensor,
    v: int,
    f: int,
    sigma: torch.Tensor,
    factors: Tuple[int, int, int],
    mean_preserve: bool,
    detach_weights: bool,
) -> torch.Tensor:
    D, H, W = volume.shape
    fz, fy, fx = factors
    patch_k = 2 * f + 1
    total_pad = v + f

    # ---- pre-pad volume once (reflect BC) ----------------------------------
    vol5 = volume.unsqueeze(0).unsqueeze(0)
    vol_big = F.pad(vol5, [total_pad] * 6, mode="reflect").squeeze(0).squeeze(0)

    vol_ctr_f = vol_big[v : v + D + 2*f,
                        v : v + H + 2*f,
                        v : v + W + 2*f]

    # ---- block means for mean-preservation ---------------------------------
    if mean_preserve:
        block_means = F.avg_pool3d(
            vol5, kernel_size=(fz, fy, fx), stride=(fz, fy, fx)
        ).squeeze(0).squeeze(0)
        block_means_full = (block_means
                            .repeat_interleave(fz, 0)
                            .repeat_interleave(fy, 1)
                            .repeat_interleave(fx, 2))

    # ---- local means for fast mean-difference rejection --------------------
    lm_pad = F.pad(
        F.avg_pool3d(
            F.pad(vol5, [1] * 6, mode="reflect"),
            kernel_size=3, stride=1, padding=0,
        ),
        [v] * 6, mode="reflect",
    ).squeeze(0).squeeze(0)
    local_mean = lm_pad[v : v + D, v : v + H, v : v + W]

    # ---- per-voxel noise² --------------------------------------------------
    h_sq = sigma.pow(2).clamp(min=1e-10)
    sigma_mask = sigma >= 1.0

    # ---- NLM accumulation --------------------------------------------------
    weighted_sum = volume.new_zeros(D, H, W)
    weight_total = volume.new_ones(D, H, W)

    for dz in range(-v, v + 1):
        for dy in range(-v, v + 1):
            for dx in range(-v, v + 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue

                sz, sy, sx = v + dz, v + dy, v + dx

                shifted_f = vol_big[sz : sz + D + 2*f,
                                    sy : sy + H + 2*f,
                                    sx : sx + W + 2*f]
                shifted   = vol_big[sz + f : sz + f + D,
                                    sy + f : sy + f + H,
                                    sx + f : sx + f + W]

                diff_sq = (vol_ctr_f - shifted_f).pow(2)
                patch_dist = F.avg_pool3d(
                    diff_sq.unsqueeze(0).unsqueeze(0),
                    kernel_size=patch_k, stride=1, padding=0
                ).squeeze(0).squeeze(0)

                shifted_lm = lm_pad[v + dz : v + dz + D,
                                    v + dy : v + dy + H,
                                    v + dx : v + dx + W]

                exponent = (patch_dist / (2.0 * h_sq) - 1.0).clamp(min=0.0)
                w = torch.exp(-exponent)
                w = w.masked_fill(
                    (local_mean - shifted_lm).abs() > 0.6 * sigma, 0.0
                )
                w = w.masked_fill(~sigma_mask, 0.0)

                if detach_weights:
                    w = w.detach()

                weighted_sum.add_(w * shifted)
                weight_total.add_(w.detach() if detach_weights else w)

    output = (volume + weighted_sum) / weight_total

    # ---- mean-preservation correction -------------------------------------
    if not mean_preserve:
        return output

    out5 = output.unsqueeze(0).unsqueeze(0)
    out_block_means = F.avg_pool3d(
        out5, kernel_size=(fz, fy, fx), stride=(fz, fy, fx)
    ).squeeze(0).squeeze(0)
    out_block_means_full = (out_block_means
                            .repeat_interleave(fz, 0)
                            .repeat_interleave(fy, 1)
                            .repeat_interleave(fx, 2))

    return output + (block_means_full - out_block_means_full)
