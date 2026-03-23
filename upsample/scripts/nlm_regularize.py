#!/usr/bin/env python3
"""
Apply a single NLM regularization step to a MINC2 volume (PyTorch backend).

This exposes ``sr_pytorch.regularize.regularize`` directly as a CLI tool,
useful for:
  * post-processing an already-interpolated HR volume
  * tuning NLM parameters (v, f, sigma) without re-running the full pipeline
  * inspecting the effect of a single regularization pass

Usage
-----
    python upsample/scripts/nlm_regularize.py input.mnc output.mnc [options]

Examples
--------
    # auto-estimated sigma, default v=3 f=1, GPU
    python upsample/scripts/nlm_regularize.py hr.mnc hr_reg.mnc

    # explicit scalar sigma, CPU only
    python upsample/scripts/nlm_regularize.py hr.mnc hr_reg.mnc --sigma 5.0 --device cpu

    # per-voxel sigma from a separate MINC file
    python upsample/scripts/nlm_regularize.py hr.mnc hr_reg.mnc --sigma-file sigma.mnc

    # factors 2 1 1 for mean-preservation of a Z-upsampled volume
    python upsample/scripts/nlm_regularize.py hr.mnc hr_reg.mnc --factors 2 1 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_UPSAMPLE = Path(__file__).resolve().parent.parent
if str(_UPSAMPLE) not in sys.path:
    sys.path.insert(0, str(_UPSAMPLE))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-pass NLM regularization of a MINC2 volume (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",  type=Path, help="Input MINC2 file (.mnc)")
    p.add_argument("output", type=Path, help="Output MINC2 file (.mnc)")

    # NLM kernel parameters
    p.add_argument("--v", type=int, default=3,
                   help="Search-neighbourhood radius")
    p.add_argument("--f", type=int, default=1,
                   help="Patch half-size (patch is (2f+1)^3 voxels)")
    p.add_argument("--factors", nargs=3, type=int, default=[1, 1, 1],
                   metavar=("FZ", "FY", "FX"),
                   help="Upsampling factors used for mean-preservation "
                        "(set to the upsampling factors if input is an "
                        "already-interpolated HR volume)")

    # Sigma / noise level
    sigma_grp = p.add_mutually_exclusive_group()
    sigma_grp.add_argument("--sigma", type=float, default=None,
                           metavar="VALUE",
                           help="Constant noise level applied to every voxel. "
                                "If omitted, sigma is auto-estimated from the "
                                "input volume using the Manjon local-std method.")
    sigma_grp.add_argument("--sigma-file", type=Path, default=None,
                           metavar="FILE",
                           help="MINC2 file containing a per-voxel noise-level "
                                "map (must have the same shape as the input).")

    # PyTorch device / dtype
    p.add_argument("--device", default=None, metavar="DEVICE",
                   help="PyTorch device (e.g. 'cpu', 'cuda', 'cuda:1'). "
                        "Defaults to 'cuda' if available, else 'cpu'.")
    p.add_argument("--dtype", choices=["float32", "float64"], default=None,
                   help="Floating-point dtype. "
                        "Defaults to float32 on CUDA, float64 on CPU.")

    p.add_argument("--verbose", action="store_true",
                   help="Print progress and parameter summary")
    return p.parse_args()


def _auto_sigma(vol_t: torch.Tensor) -> torch.Tensor:
    """Estimate per-voxel noise level (matches sr_pytorch.core._noise_level)."""
    import torch.nn.functional as F
    vol5 = vol_t.unsqueeze(0).unsqueeze(0)
    padded    = F.pad(vol5,        [1] * 6, mode="reflect")
    padded_sq = F.pad(vol5.pow(2), [1] * 6, mode="reflect")
    lmean   = F.avg_pool3d(padded,    kernel_size=3, stride=1, padding=0)
    lsqmean = F.avg_pool3d(padded_sq, kernel_size=3, stride=1, padding=0)
    sigma = (lsqmean - lmean.pow(2)).clamp(min=0).sqrt().squeeze(0).squeeze(0)
    sig5 = sigma.unsqueeze(0).unsqueeze(0)
    padded_s = torch.nn.functional.pad(sig5, [1] * 6, mode="reflect")
    smoothed = F.avg_pool3d(padded_s, kernel_size=3, stride=1, padding=0)
    return (smoothed / 2.0).squeeze(0).squeeze(0)


def main() -> None:
    args = _parse_args()

    # ---- I/O helpers -------------------------------------------------------
    try:
        from sr.minc.io import format_history, load_minc_volume_np, save_minc_volume
    except ImportError as exc:
        sys.exit(f"ERROR: Could not import MINC I/O helpers: {exc}")

    # ---- device / dtype ----------------------------------------------------
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = torch.float32 if device.type == "cuda" else torch.float64

    factors = tuple(args.factors)   # (fz, fy, fx)

    if args.verbose:
        print(f"Device  : {device}  dtype={dtype}")
        print(f"Params  : v={args.v}  f={args.f}  factors={factors}")

    # ---- load input --------------------------------------------------------
    if args.verbose:
        print(f"Loading {args.input}")
    try:
        vol_np, v2w = load_minc_volume_np(str(args.input))
    except Exception as exc:
        sys.exit(f"ERROR loading {args.input}: {exc}")

    if args.verbose:
        print(f"  shape={vol_np.shape}  min={vol_np.min():.1f}  max={vol_np.max():.1f}")

    vol_t = torch.from_numpy(vol_np).to(device=device, dtype=dtype)

    # ---- build sigma tensor ------------------------------------------------
    if args.sigma_file is not None:
        if args.verbose:
            print(f"Loading sigma from {args.sigma_file}")
        try:
            sig_np, _ = load_minc_volume_np(str(args.sigma_file))
        except Exception as exc:
            sys.exit(f"ERROR loading sigma file {args.sigma_file}: {exc}")
        if sig_np.shape != vol_np.shape:
            sys.exit(
                f"ERROR: sigma file shape {sig_np.shape} != "
                f"input shape {vol_np.shape}"
            )
        sigma_t = torch.from_numpy(sig_np).to(device=device, dtype=dtype)
        if args.verbose:
            print(f"  sigma: min={sig_np.min():.3f}  max={sig_np.max():.3f}  "
                  f"mean={sig_np.mean():.3f}")

    elif args.sigma is not None:
        sigma_t = torch.full_like(vol_t, args.sigma)
        if args.verbose:
            print(f"Sigma   : constant {args.sigma}")

    else:
        if args.verbose:
            print("Estimating sigma from input volume …")
        sigma_t = _auto_sigma(vol_t)
        sig_np_est = sigma_t.cpu().numpy()
        if args.verbose:
            print(f"  sigma: min={sig_np_est.min():.3f}  max={sig_np_est.max():.3f}  "
                  f"mean={sig_np_est.mean():.3f}")

    # ---- regularize --------------------------------------------------------
    from sr_pytorch.regularize import regularize

    if args.verbose:
        print("Running NLM regularization …")

    out_t = regularize(vol_t, args.v, args.f, sigma_t, factors)
    out_np = out_t.cpu().to(torch.float64).numpy()

    if args.verbose:
        print(f"  output: min={out_np.min():.1f}  max={out_np.max():.1f}")

    # ---- save --------------------------------------------------------------
    history = format_history(sys.argv)
    if args.verbose:
        print(f"Saving {args.output}")
    try:
        save_minc_volume(
            str(args.output),
            out_np,
            v2w,
            ref_fname=str(args.input),
            history=history,
        )
    except Exception as exc:
        sys.exit(f"ERROR saving {args.output}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
