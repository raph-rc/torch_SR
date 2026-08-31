"""Tests for the sigma-mode dispatch in upsample/denoise.py.

That dispatch (per-voxel auto / global auto / explicit global sigma) lives
in the CLI script itself, not in ``sr_pytorch.regularize`` -- see that
module's docstring, "Denoising extensions". Only ``_auto_sigma`` and the
``_sigma`` argparse type are unit-tested directly here; the rest of
``main()`` is I/O plumbing exercised manually against a real MINC file (see
the backport plan's verification step), matching how
``upsample/nlm_filter.py`` is tested.
"""

import argparse
import importlib.util
import pathlib
import sys

import pytest
import torch

_UPSAMPLE = pathlib.Path(__file__).resolve().parent.parent
if str(_UPSAMPLE) not in sys.path:
    sys.path.insert(0, str(_UPSAMPLE))

_spec = importlib.util.spec_from_file_location(
    "upsample_denoise_cli", _UPSAMPLE / "denoise.py")
assert _spec is not None and _spec.loader is not None
_denoise_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_denoise_cli)


def test_sigma_argtype_accepts_auto_and_numbers():
    assert _denoise_cli._sigma("auto") == "auto"
    assert _denoise_cli._sigma("0") == "auto"
    assert _denoise_cli._sigma("5.0") == 5.0


def test_sigma_argtype_rejects_nonpositive_numbers():
    with pytest.raises(argparse.ArgumentTypeError):
        _denoise_cli._sigma("-1.0")


def test_auto_sigma_matches_sr_pytorch_core_noise_level():
    from sr_pytorch.core import _noise_level

    torch.manual_seed(0)
    vol = torch.rand(12, 12, 12, dtype=torch.float64) * 256.0
    assert torch.allclose(_denoise_cli._auto_sigma(vol), _noise_level(vol))
