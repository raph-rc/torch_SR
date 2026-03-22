"""PSNR regression tests for initial_interpolation and nlm_upsample.

Fixtures are small 3-D blocks extracted from ``VF_20190417_MP2RAGE.mnc``
by ``scripts/generate_test_blocks.py``.  Run that script once to create
``data/psnr_factor_{2,3,5,7}.npz`` before running these tests.

Skip conditions
---------------
* C extension absent  → entire module is skipped.
* .npz file missing   → individual test is skipped with a helpful message.

Floor values
------------
Set from real measurements on the brain block (CZ=151, CY=99, CX=95)
using max_iter=3, v=3, f=1:

  factor  spline PSNR  NLM PSNR
  ------  -----------  --------
     2       25.615      27.158
     3       21.919      22.462
     5       18.823      19.034
     7       17.227      17.316

NLM floor   = observed NLM PSNR  - 1.0 dB (allows minor regressions)
Spline floor = observed spline PSNR - 2.0 dB (catches data-extraction bugs)
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
import pytest

from sr.metrics import compute_psnr
from sr.regularize import have_c_ext

# ---------------------------------------------------------------------------
# Module-level skip: C extension required for NLM tests
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not have_c_ext(),
    reason="C extension not built — run: python upsample/sr/_nlm_build.py",
)

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Floor constants (calibrated from real measurements, see module docstring)
# ---------------------------------------------------------------------------
_NLM_FLOORS: dict[int, float] = {
    2: 26.1,   # 27.158 - 1.0
    3: 21.4,   # 22.462 - 1.0
    5: 18.0,   # 19.034 - 1.0
    7: 16.3,   # 17.316 - 1.0
}

_SPLINE_FLOORS: dict[int, float] = {
    2: 23.6,   # 25.615 - 2.0
    3: 19.9,   # 21.919 - 2.0
    5: 16.8,   # 18.823 - 2.0
    7: 15.2,   # 17.227 - 2.0
}


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------

def _load_fixture(factor: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (lr_block, hr_block, factor) from the pre-generated .npz."""
    path = DATA_DIR / f"psnr_factor_{factor}.npz"
    if not path.exists():
        pytest.skip(
            f"Fixture missing: {path}\n"
            "Run:  python upsample/scripts/generate_test_blocks.py"
        )
    npz = np.load(path)
    return npz["lr"], npz["hr"], int(npz["factor"])


# ---------------------------------------------------------------------------
# Test 1 — NLM must beat spline for every factor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factor", [2, 3, 5, 7])
def test_nlm_beats_spline(factor: int) -> None:
    """NLM PSNR must exceed spline PSNR (0.05 dB grace for float scatter)."""
    from sr.core import nlm_upsample
    from sr.interpolate import initial_interpolation

    lr, hr, f = _load_fixture(factor)

    spline_hr = initial_interpolation(lr, (f, f, f))
    nlm_hr    = nlm_upsample(lr, factors=(f, f, f), max_iter=3)

    psnr_spline = compute_psnr(hr, spline_hr)
    psnr_nlm    = compute_psnr(hr, nlm_hr)

    assert psnr_nlm > psnr_spline - 0.05, (
        f"factor={f}: NLM PSNR {psnr_nlm:.3f} dB did not exceed "
        f"spline PSNR {psnr_spline:.3f} dB"
    )


# ---------------------------------------------------------------------------
# Test 2 — NLM PSNR floor (catches total regression in the C kernel)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "factor,floor_db",
    [(f, _NLM_FLOORS[f]) for f in [2, 3, 5, 7]],
)
def test_nlm_psnr_floor(factor: int, floor_db: float) -> None:
    """NLM PSNR must stay above the calibrated floor."""
    from sr.core import nlm_upsample

    lr, hr, f = _load_fixture(factor)
    nlm_hr   = nlm_upsample(lr, factors=(f, f, f), max_iter=3)
    psnr_nlm = compute_psnr(hr, nlm_hr)

    assert psnr_nlm >= floor_db, (
        f"factor={f}: NLM PSNR {psnr_nlm:.3f} dB is below floor {floor_db:.1f} dB"
    )


# ---------------------------------------------------------------------------
# Test 3 — Spline PSNR floor (catches data-extraction or interpolation bugs)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "factor,floor_db",
    [(f, _SPLINE_FLOORS[f]) for f in [2, 3, 5, 7]],
)
def test_spline_psnr_floor(factor: int, floor_db: float) -> None:
    """Spline PSNR must stay above the calibrated floor.

    Runs without NLM; verifies that the .npz fixtures and the spline
    interpolation are both correct independently of the C extension.
    (Module-level skip still applies because we import from sr.regularize.)
    """
    from sr.interpolate import initial_interpolation

    lr, hr, f = _load_fixture(factor)
    spline_hr   = initial_interpolation(lr, (f, f, f))
    psnr_spline = compute_psnr(hr, spline_hr)

    assert psnr_spline >= floor_db, (
        f"factor={f}: spline PSNR {psnr_spline:.3f} dB is below floor {floor_db:.1f} dB"
    )
