"""Shared pytest fixtures."""
import pytest
import torch


@pytest.fixture
def small_volume() -> torch.Tensor:
    """Return a small deterministic 3-D volume for testing."""
    torch.manual_seed(0)
    return torch.rand(20, 20, 20)
