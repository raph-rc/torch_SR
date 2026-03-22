#!/usr/bin/env python3
"""
NLM super-resolution upsampling for MINC2 volumes.

Usage
-----
    python upsample/upsample.py input.mnc output.mnc [options]

Example — upsample only along Z (slice direction) by a factor of 2::

    python upsample/upsample.py VF_20190417_MP2RAGE_lr.mnc out.mnc --factors 2 1 1

Reference
---------
Manjon JV, Coupe P, Buades A, Collins DL, Robles M.
MRI superresolution using self-similarity and image priors.
Int J Biomed Imaging. 2010.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NLM super-resolution upsampling (Manjon 2010)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",  type=Path, help="Input MINC2 file (.mnc)")
    p.add_argument("output", type=Path, help="Output MINC2 file (.mnc)")
    p.add_argument(
        "--factors",
        nargs=3,
        type=int,
        default=[2, 1, 1],
        metavar=("FZ", "FY", "FX"),
        help="Integer upsampling factors along (Z, Y, X) — numpy axis order",
    )
    p.add_argument(
        "--v",
        type=int,
        default=3,
        help="Search-neighbourhood radius",
    )
    p.add_argument(
        "--f",
        type=int,
        default=1,
        help="Patch half-size  (patch is (2f+1)^3 voxels)",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1.2,
        help="Convergence ratio threshold",
    )
    p.add_argument(
        "--initial-only",
        action="store_true",
        help="Skip NLM iterations; output the spline-interpolated initial estimate only",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N NLM iterations regardless of convergence (0 = run to convergence)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print iteration-level progress",
    )
    return p.parse_args()


def _adjusted_v2w(
    v2w: np.ndarray,
    factors: tuple[int, int, int],
) -> np.ndarray:
    """Return a new voxel-to-world matrix for the upsampled volume.

    MINC geometry convention:
    - ``decompose(v2w)`` → (start, step, dir_cos) where step[0..2] = [x, y, z].
    - Numpy axis order after ``setup_standard_order`` is [z, y, x].
    - ``factors`` = (fz, fy, fx) match numpy axis order.

    For each MINC dimension *i* upsampled by factor *f_i*:
    - ``new_step[i] = step[i] / f_i``
    - ``new_start[i] = start[i] - (f_i - 1) / 2 * new_step[i]``

    This places LR voxel *k* at HR position ``k * f_i + (f_i - 1) / 2``,
    consistent with ``InitialInterpolation`` in the Manjon algorithm.
    """
    from sr.minc.geo import decompose, compose

    start, step, dir_cos = decompose(v2w)
    fz, fy, fx = factors

    # MINC dim order:  index 0 = x, 1 = y, 2 = z
    # numpy dim order: index 0 = z, 1 = y, 2 = x
    minc_factors = np.array([fx, fy, fz], dtype=float)

    new_step = step / minc_factors
    new_start = start - (minc_factors - 1) / 2.0 * new_step

    return compose(new_start, new_step, dir_cos)


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- I/O imports ---------------------------------------------------
    try:
        from sr.minc.io import format_history, load_minc_volume, save_minc_volume
    except ImportError as exc:
        sys.exit(
            f"ERROR: Could not import MINC I/O helpers.\n"
            f"Install minc2_simple:  pip install -e minc2-simple/python/\n"
            f"Detail: {exc}"
        )

    # ---- C extension check --------------------------------------------
    from sr.regularize import have_c_ext
    if not have_c_ext() and not args.initial_only:
        sys.exit(
            "ERROR: C extension not built.\n"
            "Run:  python upsample/sr/_nlm_build.py"
        )

    from sr.core import nlm_upsample
    from sr.interpolate import initial_interpolation

    # ---- Load ---------------------------------------------------------
    print(f"Loading {args.input}")
    try:
        data, v2w = load_minc_volume(str(args.input))
    except Exception as exc:
        sys.exit(f"ERROR loading {args.input}: {exc}")

    # Convert torch.Tensor → numpy if minc2_simple returned a tensor.
    if hasattr(data, "numpy"):
        vol = data.numpy().astype(np.float64)
    else:
        vol = np.asarray(data, dtype=np.float64)
    v2w = np.asarray(v2w, dtype=np.float64)

    factors = tuple(args.factors)          # (fz, fy, fx)
    fz, fy, fx = factors
    print(
        f"  Input shape : {vol.shape}   (Z={vol.shape[0]}, Y={vol.shape[1]}, X={vol.shape[2]})"
    )
    print(f"  Factors     : fz={fz}  fy={fy}  fx={fx}")
    expected_hr = (vol.shape[0]*fz, vol.shape[1]*fy, vol.shape[2]*fx)
    print(f"  Output shape: {expected_hr}")

    # ---- Upsample -----------------------------------------------------
    _history = format_history(sys.argv)

    if args.initial_only:
        print("Running initial spline interpolation only …")
        hr = initial_interpolation(vol, factors)
    else:
        kw = dict(factors=factors, v=args.v, f=args.f, tol=args.tol)
        if args.max_iter > 0:
            kw["max_iter"] = args.max_iter
            print(f"Running NLM upsampling (max {args.max_iter} iterations) …")
        else:
            print("Running NLM iterative upsampling (to convergence) …")
        hr = nlm_upsample(vol, **kw)

    # ---- Adjust geometry ----------------------------------------------
    v2w_hr = _adjusted_v2w(v2w, factors)

    # ---- Save ---------------------------------------------------------
    print(f"Saving {args.output}")
    try:
        save_minc_volume(
            str(args.output),
            hr,
            np.asmatrix(v2w_hr),
            ref_fname=str(args.input),
            history=_history,
        )
    except Exception as exc:
        sys.exit(f"ERROR saving {args.output}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
