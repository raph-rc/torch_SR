#!/usr/bin/env python3
"""Generate .npz regression-test fixtures for PSNR tests.

Run once from the repository root::

    python upsample/scripts/generate_test_blocks.py

Writes ``upsample/data/psnr_factor_{f}.npz`` for factors 2, 3, 5, 7.

Each file contains:
  ``hr``     float64 ndarray — HR ground-truth block
  ``lr``     float64 ndarray — LR block produced by block-averaging
  ``factor`` 0-d int array  — the upsampling factor

Source: ``/app/VF_20190417_MP2RAGE.mnc``, shape (260, 228, 192).

Brain-centre voxel (z=151, y=99, x=95) was determined as the median
coordinate of voxels with value > 500 and is kept constant so that
generated fixtures are reproducible.

Downsampling uses block-averaging which matches the Manjon model
assumption: each LR voxel is the mean of an (f×f×f) HR sub-block.

LR block sizes are chosen so that:
  * LR side >= 2*v+1 = 7 (NLM search radius v=3)
  * Total .npz size < 5 MB

After writing the files the script also prints reference PSNR values
(spline and NLM at max_iter=3) so the developer can calibrate floor
assertions in ``tests/test_psnr.py``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap: make 'sr' importable when run from /app
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent         # upsample/scripts/
_UPSAMPLE   = _SCRIPT_DIR.parent                      # upsample/

if str(_UPSAMPLE) not in sys.path:
    sys.path.insert(0, str(_UPSAMPLE))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HR_MINC  = str(_UPSAMPLE.parent / "VF_20190417_MP2RAGE.mnc")
OUT_DIR  = _UPSAMPLE / "data"

# Brain-centre voxel (constant for reproducibility).
CZ, CY, CX = 151, 99, 95

# (factor, lr_side)  — HR side = lr_side * factor
CONFIGS: list[tuple[int, int]] = [
    (2, 12),
    (3, 12),
    (5, 10),
    (7,  8),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_average(hr: np.ndarray, f: int) -> np.ndarray:
    """Downsample *hr* by block-averaging with factor *f* in each dimension."""
    n = hr.shape[0] // f
    return hr.reshape(n, f, n, f, n, f).mean(axis=(1, 3, 5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from sr.minc.io import load_minc_volume
    except ImportError as exc:
        sys.exit(
            f"Cannot import sr.minc.io — check that minc2-simple is installed.\n{exc}"
        )

    from sr.interpolate import initial_interpolation
    from sr.core import nlm_upsample
    from sr.regularize import have_c_ext
    from sr.metrics import compute_psnr

    print(f"Loading {HR_MINC} …")
    data, _ = load_minc_volume(HR_MINC)
    arr = (
        data.numpy().astype(np.float64)
        if hasattr(data, "numpy")
        else np.asarray(data, dtype=np.float64)
    )
    print(f"  shape={arr.shape}, max={arr.max():.1f}\n")

    OUT_DIR.mkdir(exist_ok=True)

    header = f"{'factor':>8} {'LR':>6} {'HR':>6} {'PSNR spline':>14} {'PSNR NLM':>12} {'time':>8}"
    print(header)
    print("-" * len(header))

    for f, lr_n in CONFIGS:
        hr_n = lr_n * f
        h = hr_n // 2

        # Extract cubic HR block centred at (CZ, CY, CX).
        hr_block = arr[
            CZ - h : CZ + h,
            CY - h : CY + h,
            CX - h : CX + h,
        ].copy()

        if hr_block.shape != (hr_n, hr_n, hr_n):
            print(
                f"WARNING factor={f}: expected ({hr_n},{hr_n},{hr_n}), "
                f"got {hr_block.shape} — adjust CZ/CY/CX or LR side."
            )
            continue

        lr_block = _block_average(hr_block, f)
        assert lr_block.shape == (lr_n, lr_n, lr_n)

        # Save fixture.
        out_path = OUT_DIR / f"psnr_factor_{f}.npz"
        np.savez_compressed(
            out_path,
            hr=hr_block,
            lr=lr_block,
            factor=np.array(f),
        )
        size_kb = out_path.stat().st_size / 1024

        # Compute reference PSNR values for floor calibration.
        t0 = time.perf_counter()
        spline_hr = initial_interpolation(lr_block, (f, f, f))
        psnr_spline = compute_psnr(hr_block, spline_hr)

        if have_c_ext():
            nlm_hr = nlm_upsample(lr_block, factors=(f, f, f), max_iter=3)
            psnr_nlm = compute_psnr(hr_block, nlm_hr)
        else:
            psnr_nlm = float("nan")

        elapsed = time.perf_counter() - t0

        print(
            f"  {f:>6}   {lr_n:>4}^3  {hr_n:>4}^3"
            f"  {psnr_spline:>12.3f} dB"
            f"  {psnr_nlm:>10.3f} dB"
            f"  {elapsed:>6.1f}s"
            f"   ({size_kb:.0f} KB)"
        )

    total_kb = sum(
        (OUT_DIR / f"psnr_factor_{f}.npz").stat().st_size
        for f, _ in CONFIGS
        if (OUT_DIR / f"psnr_factor_{f}.npz").exists()
    ) / 1024
    print(f"\nTotal fixture size: {total_kb:.0f} KB")
    print("\nUse the PSNR values above to set floor constants in tests/test_psnr.py:")
    print("  NLM   floor = observed NLM PSNR  - 1.0 dB")
    print("  Spline floor = observed spline PSNR - 2.0 dB")


if __name__ == "__main__":
    main()
