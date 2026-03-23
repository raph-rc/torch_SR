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
) -> torch.Tensor:
    """NLM regularization – PyTorch implementation.

    Args:
        volume: Input volume, shape ``(D, H, W)``, float.
        v:      Search-neighbourhood radius.
        f:      Patch half-size; patch covers ``(2f+1)³`` voxels.
        sigma:  Per-voxel noise-level estimate, same shape as *volume*.
        factors: Upsampling factors ``(fz, fy, fx)``.

    Returns:
        Regularized volume, same shape and device as *volume*.
    """
    D, H, W = volume.shape
    fz, fy, fx = factors
    patch_k = 2 * f + 1          # kernel size for avg_pool3d patch distance
    total_pad = v + f             # single padding covers search + patch access

    # ---- pre-pad volume once (reflect BC, matching C kernel mirror BC) -----
    vol5 = volume.unsqueeze(0).unsqueeze(0)          # (1,1,D,H,W)
    vol_big = F.pad(vol5, [total_pad] * 6, mode="reflect").squeeze(0).squeeze(0)
    # shape: (D + 2*total_pad, H + 2*total_pad, W + 2*total_pad)

    # central slice with f-padding: used as the "anchor" for patch distances
    vol_ctr_f = vol_big[v : v + D + 2*f,
                        v : v + H + 2*f,
                        v : v + W + 2*f]             # (D+2f, H+2f, W+2f)

    # ---- block means of volume for mean-preservation -----------------------
    block_means = F.avg_pool3d(
        vol5, kernel_size=(fz, fy, fx), stride=(fz, fy, fx)
    ).squeeze(0).squeeze(0)
    block_means_full = (block_means
                        .repeat_interleave(fz, 0)
                        .repeat_interleave(fy, 1)
                        .repeat_interleave(fx, 2))   # (D, H, W)

    # ---- local means for fast mean-difference rejection --------------------
    # Use reflect padding (matches C kernel mirror BC, not zero-padding)
    lm_pad = F.pad(
        F.avg_pool3d(
            F.pad(vol5, [1] * 6, mode="reflect"),
            kernel_size=3, stride=1, padding=0,
        ),
        [v] * 6, mode="reflect",
    ).squeeze(0).squeeze(0)                          # (D+2v, H+2v, W+2v)
    local_mean = lm_pad[v : v + D, v : v + H, v : v + W]  # (D, H, W)

    # ---- pre-compute per-voxel noise² --------------------------------------
    h_sq = sigma.pow(2).clamp(min=1e-10)
    sigma_mask = sigma >= 1.0                         # voxels worth filtering

    # ---- NLM accumulation --------------------------------------------------
    weighted_sum = volume.new_zeros(D, H, W)
    weight_total = volume.new_ones(D, H, W)           # self-weight = 1

    for dz in range(-v, v + 1):
        for dy in range(-v, v + 1):
            for dx in range(-v, v + 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue

                sz, sy, sx = v + dz, v + dy, v + dx

                # shifted patches (D+2f, H+2f, W+2f) for patch distance
                shifted_f = vol_big[sz : sz + D + 2*f,
                                    sy : sy + H + 2*f,
                                    sx : sx + W + 2*f]

                # shifted values (D, H, W) for weighted accumulation
                shifted = vol_big[sz + f : sz + f + D,
                                  sy + f : sy + f + H,
                                  sx + f : sx + f + W]

                # patch-wise L2 distance via avg_pool3d  → (D, H, W)
                diff_sq = (vol_ctr_f - shifted_f).pow(2)
                patch_dist = F.avg_pool3d(
                    diff_sq.unsqueeze(0).unsqueeze(0),
                    kernel_size=patch_k, stride=1, padding=0
                ).squeeze(0).squeeze(0)

                # local mean of the shifted position
                shifted_lm = lm_pad[v + dz : v + dz + D,
                                    v + dy : v + dy + H,
                                    v + dx : v + dx + W]

                # NLM weight: exp(-max(d/(2σ²) − 1, 0))
                exponent = (patch_dist / (2.0 * h_sq) - 1.0).clamp(min=0.0)
                w = torch.exp(-exponent)

                # fast rejection + low-sigma mask
                w = w.masked_fill(
                    (local_mean - shifted_lm).abs() > 0.6 * sigma, 0.0
                )
                w = w.masked_fill(~sigma_mask, 0.0)

                weighted_sum.add_(w * shifted)
                weight_total.add_(w)

    output = (volume + weighted_sum) / weight_total

    # ---- mean-preservation correction -------------------------------------
    out5 = output.unsqueeze(0).unsqueeze(0)
    out_block_means = F.avg_pool3d(
        out5, kernel_size=(fz, fy, fx), stride=(fz, fy, fx)
    ).squeeze(0).squeeze(0)
    out_block_means_full = (out_block_means
                            .repeat_interleave(fz, 0)
                            .repeat_interleave(fy, 1)
                            .repeat_interleave(fx, 2))

    return output + (block_means_full - out_block_means_full)
