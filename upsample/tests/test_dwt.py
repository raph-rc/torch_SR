"""Property tests for sr_pytorch.dwt (one-level forward D4 wavelet transform).

Ported from torch_n3's ``tests/test_dwt.py``, minus the comparison against
``torch_n3.backends.legacy`` (the vendored EZminc C++ oracle): torch_SR has
no such oracle, so only self-consistency properties are asserted here. See
``upsample/sr_pytorch/dwt.py``'s module docstring for what the transform is
supposed to compute.
"""

import pytest
import torch

from sr_pytorch.dwt import dwt_forward


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
