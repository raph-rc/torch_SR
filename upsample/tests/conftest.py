"""Shared pytest fixtures."""
import os

import pytest
import torch

from sr.minc.io import load_minc_volume_np
from tests import reference

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


@pytest.fixture
def small_volume() -> torch.Tensor:
    """Return a small deterministic 3-D volume for testing."""
    torch.manual_seed(0)
    return torch.rand(20, 20, 20)


def legacy_data(name):
    """A test volume, by the name the EZminc reference generation knows it as."""
    return os.path.join(DATA, name)


def relative_rms(result, expected):
    """RMS difference over mean signal -- the suite's measure of whole-volume
    agreement. Ported from ``torch_n3/tests/conftest.py``."""
    result = torch.as_tensor(result, dtype=torch.float64)
    expected = torch.as_tensor(expected, dtype=torch.float64)
    return float(((result - expected) ** 2).mean().sqrt() / expected.mean())


@pytest.fixture(scope="session")
def legacy_output():
    """What the installed EZminc programs answered, recorded once."""
    return reference.load()


@pytest.fixture(scope="session")
def chunk():
    """The small 91x52x50 volume, fast enough for per-block tests."""
    data, _ = load_minc_volume_np(legacy_data("chunk.mnc"))
    return torch.as_tensor(data)


@pytest.fixture(scope="session")
def chunk_mask():
    data, _ = load_minc_volume_np(legacy_data("chunk_mask.mnc"))
    return torch.as_tensor(data)
