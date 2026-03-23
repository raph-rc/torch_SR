"""Tests for the PyTorch NLM upsampling implementation (sr_pytorch).

Structure mirrors test_psnr.py / test_core.py but exercises sr_pytorch.
Tests are parametrised over CPU and (optionally) CUDA devices.

Skip conditions
---------------
* .npz fixture missing   → individual PSNR test is skipped.
* CUDA unavailable       → 'cuda' device variant is skipped.

PSNR floors
-----------
Set to C-reference floor − 0.5 dB to allow for float32/float64 and
minor algorithmic differences (boundary handling, no race-condition noise).
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
import pytest
import torch

from sr.metrics import compute_psnr

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Devices to test
# ---------------------------------------------------------------------------
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")

# ---------------------------------------------------------------------------
# PSNR floors: C-reference measured values − 1.5 dB (generous margin that
# still catches total regressions in the PyTorch kernel)
# ---------------------------------------------------------------------------
_PT_NLM_FLOORS: dict[int, float] = {
    2: 25.6,   # 27.158 - 1.5
    3: 20.9,   # 22.462 - 1.5
    5: 17.5,   # 19.034 - 1.5
    7: 15.8,   # 17.316 - 1.5
}

_PT_SPLINE_FLOORS: dict[int, float] = {
    2: 23.6,
    3: 19.9,
    5: 16.8,
    7: 15.2,
}


def _load_fixture(factor: int) -> Tuple[np.ndarray, np.ndarray, int]:
    path = DATA_DIR / f"psnr_factor_{factor}.npz"
    if not path.exists():
        pytest.skip(
            f"Fixture missing: {path}\n"
            "Run:  python upsample/scripts/generate_test_blocks.py"
        )
    npz = np.load(path)
    return npz["lr"], npz["hr"], int(npz["factor"])


# ---------------------------------------------------------------------------
# Test 1 — output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("factors", [(2, 2, 2), (2, 1, 1), (3, 2, 1)])
def test_output_shape(factors, device):
    from sr_pytorch.core import nlm_upsample

    lr = np.random.rand(8, 10, 12).astype(np.float64) * 100 + 50
    hr = nlm_upsample(lr, factors=factors, max_iter=1, device=device)
    expected = tuple(s * f for s, f in zip(lr.shape, factors))
    assert hr.shape == expected


# ---------------------------------------------------------------------------
# Test 2 — no NaN / Inf
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", _DEVICES)
def test_no_nan_inf(device):
    from sr_pytorch.core import nlm_upsample

    lr = np.random.rand(8, 8, 8).astype(np.float64) * 200 + 10
    hr = nlm_upsample(lr, factors=(2, 2, 2), max_iter=2, device=device)
    assert np.all(np.isfinite(hr))


# ---------------------------------------------------------------------------
# Test 3 — uniform input stays (nearly) uniform
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", _DEVICES)
def test_uniform_input(device):
    from sr_pytorch.core import nlm_upsample

    lr = np.full((6, 6, 6), 128.0, dtype=np.float64)
    hr = nlm_upsample(lr, factors=(2, 2, 2), max_iter=2, device=device)
    assert np.allclose(hr, 128.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 4 — PSNR close to C reference  (max_iter=3, factors isotropic)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factor", [2, 3, 5, 7])
@pytest.mark.parametrize("device", _DEVICES)
def test_psnr_close_to_c(factor: int, device: str) -> None:
    """PyTorch PSNR must be within 0.5 dB of the C reference."""
    from sr.core import nlm_upsample as nlm_c
    from sr_pytorch.core import nlm_upsample as nlm_t
    from sr.regularize import have_c_ext

    if not have_c_ext():
        pytest.skip("C extension not built")

    lr, hr, f = _load_fixture(factor)
    dtype = torch.float32 if device == "cuda" else torch.float64

    psnr_c = compute_psnr(hr, nlm_c(lr, factors=(f, f, f), max_iter=3))
    psnr_t = compute_psnr(hr, nlm_t(lr, factors=(f, f, f), max_iter=3,
                                     device=device, dtype=dtype))

    assert abs(psnr_t - psnr_c) < 0.5, (
        f"factor={f} device={device}: "
        f"PyTorch PSNR {psnr_t:.3f} dB vs C {psnr_c:.3f} dB "
        f"(diff {psnr_t - psnr_c:+.3f} dB)"
    )


# ---------------------------------------------------------------------------
# Test 5 — absolute PSNR floor (catches total regression in PyTorch kernel)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "factor,floor_db",
    [(f, _PT_NLM_FLOORS[f]) for f in [2, 3, 5, 7]],
)
@pytest.mark.parametrize("device", _DEVICES)
def test_nlm_psnr_floor(factor: int, floor_db: float, device: str) -> None:
    from sr_pytorch.core import nlm_upsample

    lr, hr, f = _load_fixture(factor)
    dtype = torch.float32 if device == "cuda" else torch.float64
    out = nlm_upsample(lr, factors=(f, f, f), max_iter=3,
                       device=device, dtype=dtype)
    psnr = compute_psnr(hr, out)
    assert psnr >= floor_db, (
        f"factor={f} device={device}: "
        f"PSNR {psnr:.3f} dB below floor {floor_db:.1f} dB"
    )
