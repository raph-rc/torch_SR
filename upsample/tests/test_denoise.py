"""Tests for the argument handling and sigma dispatch in upsample/denoise.py.

That dispatch (uniform auto / per-voxel auto / explicit uniform sigma /
sigma map) lives in the CLI script itself, not in ``sr_pytorch.regularize``
-- see that module's docstring, "Denoising extensions". The parser,
``_build_sigma`` and ``_resolve_mean_preserve`` are unit-tested directly
here; the rest of ``main()`` is I/O plumbing exercised manually against a
real MINC file.
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

_IO = ["in.mnc", "out.mnc"]


@pytest.fixture
def volume():
    torch.manual_seed(0)
    return torch.rand(12, 12, 12, dtype=torch.float64) * 256.0


def _args(*extra):
    return _denoise_cli._parse_args(_IO + list(extra))


# ---------------------------------------------------------------------------
# --sigma argparse type
# ---------------------------------------------------------------------------

def test_sigma_argtype_accepts_positive_numbers():
    assert _denoise_cli._sigma("5.0") == 5.0


@pytest.mark.parametrize("value", ["auto", "0", "-1.0"])
def test_sigma_argtype_rejects_non_positive_and_auto(value):
    # 'auto' was the old mincnlm-style spelling; it is gone, replaced by
    # --auto-sigma-global.
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        _denoise_cli._sigma(value)


# ---------------------------------------------------------------------------
# sigma dispatch
# ---------------------------------------------------------------------------

def test_default_sigma_is_the_uniform_auto_estimate(volume):
    default = _denoise_cli._build_sigma(_args(), volume, volume.shape)
    explicit = _denoise_cli._build_sigma(
        _args("--auto-sigma-global"), volume, volume.shape)

    assert torch.equal(default, explicit)
    assert default.unique().numel() == 1        # uniform


def test_auto_sigma_voxel_matches_core_noise_level(volume):
    from sr_pytorch.core import _noise_level

    sigma = _denoise_cli._build_sigma(
        _args("--auto-sigma-voxel"), volume, volume.shape)

    assert torch.allclose(sigma, _noise_level(volume))
    assert sigma.unique().numel() > 1            # genuinely per-voxel


def test_explicit_sigma_is_uniform(volume):
    sigma = _denoise_cli._build_sigma(
        _args("--sigma", "5.0"), volume, volume.shape)
    assert torch.equal(sigma, torch.full_like(volume, 5.0))


def test_gauss_changes_the_uniform_estimate(volume):
    rician = _denoise_cli._build_sigma(
        _args("--auto-sigma-global"), volume, volume.shape)
    gaussian = _denoise_cli._build_sigma(
        _args("--auto-sigma-global", "--gauss"), volume, volume.shape)
    assert not torch.equal(rician, gaussian)


def test_gauss_without_auto_sigma_global_is_rejected():
    with pytest.raises(SystemExit):
        _args("--gauss")


@pytest.mark.parametrize("flags", [
    ("--auto-sigma-global", "--auto-sigma-voxel"),
    ("--auto-sigma-global", "--sigma", "5.0"),
    ("--auto-sigma-voxel", "--sigma-file", "s.mnc"),
    ("--sigma", "5.0", "--sigma-file", "s.mnc"),
])
def test_sigma_flags_are_mutually_exclusive(flags):
    with pytest.raises(SystemExit):
        _args(*flags)


# ---------------------------------------------------------------------------
# --strength / --h
# ---------------------------------------------------------------------------

def test_strength_defaults_to_one():
    assert _args().strength == 1.0


def test_h_is_an_alias_for_strength():
    assert _args("--h", "2.0").strength == 2.0


def test_h_and_strength_together_are_rejected():
    with pytest.raises(SystemExit):
        _args("--h", "2.0", "--strength", "2.0")


# ---------------------------------------------------------------------------
# --factors / mean preservation
# ---------------------------------------------------------------------------

def test_factors_default_to_unity_and_mean_preserve_is_unset():
    args = _args()
    assert args.factors == [1, 1, 1]
    assert args.mean_preserve is None


@pytest.mark.parametrize("mean_preserve, factors, expected", [
    (None,  (1, 1, 1), False),
    (None,  (2, 1, 1), True),
    (False, (2, 1, 1), False),
    (True,  (2, 1, 1), True),
])
def test_resolve_mean_preserve(mean_preserve, factors, expected):
    assert _denoise_cli._resolve_mean_preserve(mean_preserve, factors) is expected


def test_mean_preserve_with_unit_factors_is_rejected():
    # It would be an exact identity -- see _resolve_mean_preserve's docstring.
    with pytest.raises(SystemExit):
        _args("--mean-preserve")


def test_defaults_filter_while_mean_preserve_would_not():
    """The resolved defaults must do real work, not reproduce the input."""
    from sr_pytorch.regularize import regularize

    # A smooth ramp plus noise, so patches actually match each other and the
    # NLM weights are non-zero.
    torch.manual_seed(0)
    ramp = torch.linspace(0.0, 100.0, 12, dtype=torch.float64)
    vol = (ramp.view(-1, 1, 1).expand(12, 12, 12).contiguous()
           + torch.rand(12, 12, 12, dtype=torch.float64) * 10.0)
    sigma = torch.full_like(vol, 5.0)

    filtered = regularize(vol, 2, 1, sigma, factors=(1, 1, 1),
                          mean_preserve=False)
    preserved = regularize(vol, 2, 1, sigma, factors=(1, 1, 1),
                           mean_preserve=True)

    assert not torch.allclose(filtered, vol)
    assert torch.allclose(preserved, vol)
