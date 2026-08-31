"""Property tests for sr_pytorch.dwt (one-level forward D4 wavelet transform),
plus a direct comparison against the vendored EZminc oracle (``_legacy``,
backported alongside these tests -- see ``upsample/_legacy/ezminc/README.md``
and ``tests/test_ezminc_backend.py``). See ``upsample/sr_pytorch/dwt.py``'s
module docstring for what the transform is supposed to compute.
"""

import pytest
import torch

from _legacy import legacy_ezminc
from sr_pytorch.dwt import dwt_forward

_requires_legacy_ext = pytest.mark.skipif(
    not legacy_ezminc.have_legacy_ext(),
    reason="EZminc extensions not built -- run: python _legacy/build_legacy.py")


@pytest.mark.parametrize("shape", [(8, 8, 8), (5, 9, 13), (16, 12, 20)])
def test_subband_count_and_shape(shape):
    vol = torch.rand(shape, dtype=torch.float64)
    subs = dwt_forward(vol)
    assert len(subs) == 8
    expected = tuple((s + 1) // 2 for s in shape)
    for sub in subs:
        assert sub.shape == expected


def test_zero_input_gives_zero_subbands():
    vol = torch.zeros(10, 10, 10, dtype=torch.float64)
    subs = dwt_forward(vol)
    for sub in subs:
        assert torch.allclose(sub, torch.zeros_like(sub))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_propagation(dtype):
    vol = torch.rand(12, 12, 12, dtype=dtype)
    subs = dwt_forward(vol)
    for sub in subs:
        assert sub.dtype == dtype


def test_finite_output_on_random_input():
    vol = torch.randn(9, 11, 7, dtype=torch.float64) * 100.0
    subs = dwt_forward(vol)
    for sub in subs:
        assert torch.isfinite(sub).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_parity_cpu_cuda():
    torch.manual_seed(0)
    vol_cpu = torch.rand(10, 10, 10, dtype=torch.float64)
    vol_cuda = vol_cpu.to("cuda")
    subs_cpu = dwt_forward(vol_cpu)
    subs_cuda = dwt_forward(vol_cuda)
    for a, b in zip(subs_cpu, subs_cuda):
        assert torch.allclose(a, b.cpu(), atol=1e-8)


def _max_rel(result, expected):
    """Largest elementwise relative difference, against the largest value.

    The float32 C++ and the float64 port agree to float32 rounding, which is
    proportional to the magnitude of the *input* (a linear transform), not
    to each output element; a subband element near zero can therefore carry
    an absolute error that dwarfs itself. Comparing the largest difference
    to the largest value in the subband measures agreement on the scale
    that matters. Ported from ``torch_n3/tests/test_dwt.py``.
    """
    result = torch.as_tensor(result, dtype=torch.float64)
    expected = torch.as_tensor(expected, dtype=torch.float64)
    return float((result - expected).abs().max() / expected.abs().max())


@_requires_legacy_ext
def test_subbands_match_legacy(chunk):
    """All eight subbands of ``chunk`` agree with the C++ transform."""
    torch_sub = dwt_forward(chunk)
    legacy_sub = legacy_ezminc.dwt_forward(chunk)
    assert len(torch_sub) == len(legacy_sub) == 8
    for j, (got, want) in enumerate(zip(torch_sub, legacy_sub)):
        assert got.shape == want.shape
        assert got.shape == tuple((d + 1) // 2 for d in chunk.shape), j
        assert _max_rel(got, want) < 1e-5


@_requires_legacy_ext
@pytest.mark.parametrize("shape", [(8, 8, 8), (7, 9, 5), (16, 3, 3),
                                   (13, 13, 13), (4, 5, 6)])
def test_subbands_match_legacy_shapes(shape):
    """Exact power-of-two, odd, and mixed extents: pad and extraction agree."""
    vol = torch.randn(shape, generator=torch.Generator().manual_seed(0)) * 1000
    for got, want in zip(dwt_forward(vol), legacy_ezminc.dwt_forward(vol)):
        assert got.shape == want.shape
        assert _max_rel(got, want) < 1e-5
