#!/usr/bin/env python3
"""Filter a MINC2 volume with one non-local-means pass (pure denoising).

    python upsample/denoise.py input.mnc output.mnc [options]

This is a plain per-voxel NLM filter, not the SR upsampling pipeline: it
calls ``sr_pytorch.regularize.regularize`` with ``factors=(1, 1, 1)`` and
``mean_preserve=False``, so there is no block-mean correction to disable.
Backported from ``torch_n3/denoise_cli.py``, restructured to reuse this
project's own conventions (``sr.minc.io`` for I/O, the same CLI shape as
``upsample/nlm_filter.py``) instead of torch_n3's ``Volume`` class.

Sigma modes (``--sigma``)
--------------------------
Mirrors mincnlm's convention, same as ``torch_n3.blocks.denoise.denoise``:

* omitted (default) — a per-voxel estimate (``sr_pytorch.core``'s
  ``_noise_level``: local std, smoothed, halved), the same auto-estimate
  ``nlm_filter.py`` uses.
* ``auto`` (or ``0``) — one global level from
  ``sr_pytorch.noise_estimate.noise_estimate(volume)[0]`` (Rician/Koay by
  default, ``--gauss`` for the Gaussian variant).
* a number — that global level, on the volume's own intensity scale.

This sigma-mode dispatch lives here, in the CLI, not inside
``regularize()`` — see ``sr_pytorch/regularize.py``'s module docstring.
There is no per-voxel ``--mask`` support for the ``auto`` global estimate
in this first pass, even though ``noise_estimate`` itself accepts one; use
``upsample/noise_estimate.py --mask`` to get a masked estimate and pass it
back in with an explicit ``--sigma <value>``.

Examples
--------
    # auto per-voxel sigma, default v=3 f=1, GPU
    python upsample/denoise.py hr.mnc hr_denoised.mnc

    # global auto-estimated sigma (Rician), doubled
    python upsample/denoise.py hr.mnc hr_denoised.mnc --sigma auto --strength 2.0

    # explicit global sigma, CPU, affine-equivariant floor
    python upsample/denoise.py hr.mnc hr_denoised.mnc --sigma 5.0 --device cpu \\
        --floor-mode scale
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_UPSAMPLE = Path(__file__).resolve().parent
if str(_UPSAMPLE) not in sys.path:
    sys.path.insert(0, str(_UPSAMPLE))


def _sigma(value: str):
    """``--sigma`` takes one global noise level or the word for not having
    one: 'auto' (or 0) estimates it from the volume, as mincnlm's ``-sigma
    0``; a number uses it as given, on the volume's own scale."""
    if value.lower() in ("auto", "0"):
        return "auto"
    level = float(value)
    if not level > 0:
        raise argparse.ArgumentTypeError(
            "a global noise level is positive; 'auto' (or 0) estimates it")
    return level


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-pass non-local-means denoising of a MINC2 "
                    "volume (Manjon et al. 2010).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Input MINC2 file (.mnc)")
    p.add_argument("output", type=Path, help="Output MINC2 file (.mnc)")

    p.add_argument("--v", type=int, default=3,
                   help="Search-neighbourhood radius; cost is cubic in this "
                        "and in nothing else")
    p.add_argument("--f", type=int, default=1,
                   help="Patch half-size (patch is (2f+1)^3 voxels)")
    p.add_argument("--strength", type=float, default=1.0,
                   help="Multiplies the estimated noise level; 0 is exactly "
                        "the identity")
    p.add_argument("--floor-mode", choices=["legacy", "scale"],
                   default="legacy",
                   help="'legacy' matches sr_pytorch.regularize's original "
                        "fixed noise floor; 'scale' derives it from the "
                        "volume's own dynamic range (affine-equivariant)")

    sigma_grp = p.add_mutually_exclusive_group()
    sigma_grp.add_argument("--sigma", type=_sigma, default=None,
                           metavar="VALUE",
                           help="use one global noise level instead of the "
                                "per-voxel estimate: 'auto' (or 0) "
                                "estimates it from the volume, a number "
                                "uses it as given")
    sigma_grp.add_argument("--sigma-file", type=Path, default=None,
                           metavar="FILE",
                           help="MINC2 file with a per-voxel noise-level "
                                "map (must match the input's shape)")

    p.add_argument("--gauss", action="store_true",
                   help="the automatic global level's noise model: Rician "
                        "(Koay-corrected) by default, Gaussian with this "
                        "flag")
    p.add_argument("--device", default=None, metavar="DEVICE",
                   help="PyTorch device, e.g. 'cuda'. Defaults to 'cuda' "
                        "if available, else 'cpu'")
    p.add_argument("--dtype", choices=["float32", "float64"], default=None,
                   help="Floating-point dtype. Defaults to float32 on "
                        "CUDA, float64 on CPU")
    p.add_argument("--verbose", action="store_true",
                   help="Print progress and parameter summary")
    return p.parse_args()


def _auto_sigma(vol_t: torch.Tensor) -> torch.Tensor:
    """Per-voxel noise level (matches sr_pytorch.core._noise_level)."""
    vol5 = vol_t.unsqueeze(0).unsqueeze(0)
    padded = F.pad(vol5, [1] * 6, mode="reflect")
    padded_sq = F.pad(vol5.pow(2), [1] * 6, mode="reflect")
    lmean = F.avg_pool3d(padded, kernel_size=3, stride=1, padding=0)
    lsqmean = F.avg_pool3d(padded_sq, kernel_size=3, stride=1, padding=0)
    sigma = (lsqmean - lmean.pow(2)).clamp(min=0).sqrt().squeeze(0).squeeze(0)
    sig5 = sigma.unsqueeze(0).unsqueeze(0)
    padded_s = F.pad(sig5, [1] * 6, mode="reflect")
    smoothed = F.avg_pool3d(padded_s, kernel_size=3, stride=1, padding=0)
    return (smoothed / 2.0).squeeze(0).squeeze(0)


def main() -> None:
    args = _parse_args()

    try:
        from sr.minc.io import format_history, load_minc_volume_np, save_minc_volume
    except ImportError as exc:
        sys.exit(f"ERROR: Could not import MINC I/O helpers: {exc}")

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = torch.float32 if device.type == "cuda" else torch.float64

    if args.verbose:
        print(f"Device  : {device}  dtype={dtype}")
        print(f"Params  : v={args.v}  f={args.f}  strength={args.strength}  "
              f"floor_mode={args.floor_mode}")

    if args.verbose:
        print(f"Loading {args.input}")
    try:
        vol_np, v2w = load_minc_volume_np(str(args.input))
    except Exception as exc:
        sys.exit(f"ERROR loading {args.input}: {exc}")

    if args.verbose:
        print(f"  shape={vol_np.shape}  min={vol_np.min():.1f}  max={vol_np.max():.1f}")

    vol_t = torch.from_numpy(vol_np).to(device=device, dtype=dtype)

    # ---- sigma-mode dispatch: resolve whatever the user asked for into a
    # per-voxel tensor here, in the CLI, before calling regularize() ---------
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

    elif args.sigma is None:
        if args.verbose:
            print("Estimating per-voxel sigma from input volume ...")
        sigma_t = _auto_sigma(vol_t)

    else:
        if args.sigma == "auto":
            from sr_pytorch.noise_estimate import noise_estimate
            if args.verbose:
                print("Estimating global sigma (noise_estimate) ...")
            level, mean_signal = noise_estimate(
                vol_t.to(torch.float64), gaussian=args.gauss)
            if args.verbose:
                print(f"  global sigma={level:.4f}  mean_signal={mean_signal:.4f}")
        else:
            level = args.sigma
        sigma_t = torch.full_like(vol_t, float(level))

    if args.verbose:
        sig_np_report = sigma_t.detach().cpu().numpy()
        print(f"  sigma: min={sig_np_report.min():.3f}  "
              f"max={sig_np_report.max():.3f}  mean={sig_np_report.mean():.3f}")

    from sr_pytorch.regularize import regularize

    if args.verbose:
        print("Running NLM denoising ...")

    start = time.time()
    out_t = regularize(vol_t, args.v, args.f, sigma_t, factors=(1, 1, 1),
                       mean_preserve=False, strength=args.strength,
                       floor_mode=args.floor_mode)
    elapsed = time.time() - start
    out_np = out_t.cpu().to(torch.float64).numpy()

    if args.verbose:
        print(f"  output: min={out_np.min():.1f}  max={out_np.max():.1f}")
        print(f"  elapsed: {elapsed:.2f}s")

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
