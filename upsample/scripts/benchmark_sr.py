#!/usr/bin/env python3
"""Benchmark C-backed vs PyTorch NLM upsampling.

Runs both implementations on every pre-generated test fixture
(``data/psnr_factor_{2,3,5,7}.npz``) and prints a comparison table of
reconstruction quality (PSNR vs ground truth) and wall-clock time.

Usage::

    # from /app/upsample
    python scripts/benchmark_sr.py

    # restrict to specific factors
    python scripts/benchmark_sr.py --factors 2 3

    # force CPU-only for PyTorch
    python scripts/benchmark_sr.py --no-cuda

    # more NLM iterations (default 3)
    python scripts/benchmark_sr.py --max-iter 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_UPSAMPLE = Path(__file__).resolve().parent.parent
if str(_UPSAMPLE) not in sys.path:
    sys.path.insert(0, str(_UPSAMPLE))

from sr.metrics import compute_psnr


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_c(lr: np.ndarray, factor: int, max_iter: int) -> tuple[np.ndarray, float]:
    from sr.core import nlm_upsample
    t0 = time.perf_counter()
    out = nlm_upsample(lr, factors=(factor, factor, factor), max_iter=max_iter)
    return out, time.perf_counter() - t0


def run_torch(
    lr: np.ndarray,
    factor: int,
    max_iter: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[np.ndarray, float]:
    from sr_pytorch.core import nlm_upsample
    # warm-up: first call may include JIT / kernel compilation overhead
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = nlm_upsample(
        lr,
        factors=(factor, factor, factor),
        max_iter=max_iter,
        device=device,
        dtype=dtype,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark C vs PyTorch NLM upsampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--factors", nargs="+", type=int, default=[2, 3, 5, 7],
        metavar="F", help="Upsampling factors to test",
    )
    p.add_argument(
        "--max-iter", type=int, default=3,
        help="NLM iterations per run",
    )
    p.add_argument(
        "--no-cuda", action="store_true",
        help="Disable CUDA even if available",
    )
    p.add_argument(
        "--no-c", action="store_true",
        help="Skip C-backed implementation",
    )
    p.add_argument(
        "--dtype", choices=["float32", "float64"], default=None,
        help="PyTorch dtype (default: float32 on CUDA, float64 on CPU)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = _UPSAMPLE / "data"

    # ---- decide which backends to run ------------------------------------
    from sr.regularize import have_c_ext
    run_c_backend = have_c_ext() and not args.no_c

    pt_devices: list[str] = ["cpu"]
    if torch.cuda.is_available() and not args.no_cuda:
        pt_devices.append("cuda")

    pt_dtype_override: torch.dtype | None = None
    if args.dtype == "float32":
        pt_dtype_override = torch.float32
    elif args.dtype == "float64":
        pt_dtype_override = torch.float64

    # ---- table header ----------------------------------------------------
    cols: list[str] = []
    if run_c_backend:
        cols += ["C PSNR", "C time"]
    for dev in pt_devices:
        label = f"PT-{dev}"
        cols += [f"{label} PSNR", f"{label} time", f"{label} ΔPSNR"]

    header_parts = [f"{'factor':>7}"] + [f"{c:>14}" for c in cols]
    header = "  ".join(header_parts)
    print(f"\n  max_iter={args.max_iter}  "
          f"devices: {['C'] * run_c_backend + ['PT-' + d for d in pt_devices]}\n")
    print(header)
    print("─" * len(header))

    # ---- per-factor loop -------------------------------------------------
    for factor in args.factors:
        path = data_dir / f"psnr_factor_{factor}.npz"
        if not path.exists():
            print(f"  factor={factor}: fixture missing — run scripts/generate_test_blocks.py")
            continue

        npz = np.load(path)
        lr: np.ndarray = npz["lr"]
        hr: np.ndarray = npz["hr"]

        row = [f"{factor:>7}"]

        # C backend
        psnr_c = float("nan")
        if run_c_backend:
            out_c, t_c = run_c(lr, factor, args.max_iter)
            psnr_c = compute_psnr(hr, out_c)
            row += [f"{psnr_c:>14.3f}", f"{t_c:>13.3f}s"]

        # PyTorch backends
        for dev in pt_devices:
            dtype = pt_dtype_override
            if dtype is None:
                dtype = torch.float32 if dev == "cuda" else torch.float64
            out_t, t_t = run_torch(lr, factor, args.max_iter, dev, dtype)
            psnr_t = compute_psnr(hr, out_t)
            delta = psnr_t - psnr_c if not np.isnan(psnr_c) else float("nan")
            row += [f"{psnr_t:>14.3f}", f"{t_t:>13.3f}s", f"{delta:>+14.3f}"]

        print("  ".join(row))

    print()


if __name__ == "__main__":
    main()
