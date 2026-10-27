#!/usr/bin/env python3
"""
CLI: NLM super-resolution upsampling for MINC2 volumes.

Usage
-----
    python scripts/nlm_upsample.py input.mnc output.mnc [options]

The script loads a MINC2 volume, applies the Manjon (2010) NLM upsampling
algorithm, and writes the result as a new MINC2 file.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NLM super-resolution upsampling (Manjon 2010)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Input file (.nii, .nii.gz)")
    p.add_argument("output", type=Path, help="Output file (.nii, .nii.gz)")
    p.add_argument(
        "--factors",
        nargs=3,
        type=int,
        default=[2, 2, 2],
        metavar=("FZ", "FY", "FX"),
        help="Integer upsampling factors along (Z, Y, X)",
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
        help="Patch half-size (patch is (2f+1)^3 voxels)",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1.2,
        help="Convergence ratio threshold",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    level = logging.WARNING
    if args.verbose:
        level = logging.INFO
    if args.debug:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Import here so missing MINC library gives a clear error at runtime.
    try:
        import nibabel as nib
    except ImportError as exc:
        sys.exit(
            f"nibabel not available — install the nibabel"
            f"Python package.\n({exc})"
        )

    from sr.core import nlm_upsample

    # _history = format_history(sys.argv)
    factors = tuple(args.factors)

    logging.getLogger().info("Loading %s", args.input)
    try:
        img = nib.load(str(args.input))
        # data, v2w = load_minc_volume(str(args.input))
    except Exception as exc:
        sys.exit(f"Failed to load {args.input}: {exc}")

    if not isinstance(img, nib.Nifti1Image):
        raise RuntimeError(f"expected nifti image, got {type(img)} instead.")

    # if isinstance(data, np.ndarray):
    #     vol = data.astype(np.float64)
    # else:
    #     # torch.Tensor from load_minc_volume
    #     vol = data.numpy().astype(np.float64)
    #
    vol = img.get_fdata()
    aff = img.affine

    logging.getLogger().info("Input shape %s, factors %s", vol.shape, factors)
    hr = nlm_upsample(vol, factors=factors, v=args.v, f=args.f, tol=args.tol)

    # Adjust voxel-to-world matrix: scale voxel sizes by 1/factor.
    import numpy.matlib  # noqa: F401 — ensure np.asmatrix works

    aff_hr = np.asmatrix(np.array(aff, dtype=np.float64))
    for d, fac in enumerate(factors):
        # Column d of the rotation/scale part is divided by fac
        # because voxel spacing shrinks by that factor
        aff_hr[0:3, d] /= fac

    logging.getLogger().info("Saving %s", args.output)
    try:
        img_hr = nib.Nifti1Image(hr, aff_hr, img.header)
        nib.save(img_hr, str(args.output))
        # save_minc_volume(
        #     str(args.output),
        #     hr,
        #     aff_hr,
        #     ref_fname=str(args.input),
        #     history=_history,
        # )
    except Exception as exc:
        sys.exit(f"Failed to save {args.output}: {exc}")

    print(f"Saved upsampled volume to {args.output}")


if __name__ == "__main__":
    main()
