# PyTorch Re-implementation of Non-local MRI upsampling (Manjon 2010)

Guidelines for AI coding agents working on this PyTorch re-implementation.

## Project Overview

This project re-implements the Manjon 2010 MRI upsampling algorithm in PyTorch.
The original algorithm (see `legacy/`) was written in C (`cMRegularizar.c`) and Matlab.

Key reference files:
- `minc2-simple/USAGE.md` — complete MINC2 I/O API reference
- `minc2-simple/example/python/minc/io.py` — I/O helper functions to reuse
- `minc2-simple/example/python/resample_pytorch_nl.py` — PyTorch spatial transform patterns

## Directory Structure

```
upsample/                    Project root — run all commands from here
upsample/requirements.txt    pip dependencies (torch, numpy, scipy, cffi)
upsample/setup.py

upsample/sr/                 Python package (import as `sr`)
  __init__.py                re-exports nlm_upsample
  exceptions.py              UpsampleError base class
  nlm_core.c                 C implementation (MEX-free port of cMRegularizarNLM3D_V2.c)
  nlm_core.h                 Public C header
  _nlm_build.py              cffi out-of-line build script (run once)
  regularize.py              Python wrapper: regularize(volume, v, f, sigma, factors)
  interpolate.py             initial_interpolation(lr_vol, factors) — scipy spline
  core.py                    nlm_upsample() — full pipeline

upsample/tests/              pytest suite
  conftest.py                shared fixtures (small_volume)
  test_interpolate.py        tests for sr.interpolate (no C ext needed)
  test_regularize.py         tests for sr.regularize  (needs C ext)
  test_core.py               integration tests         (needs C ext)

upsample/data/               sample .mnc volumes for testing (not committed)
upsample/scripts/
  nlm_upsample.py            CLI: input.mnc output.mnc [options]

legacy/                      Original Matlab/C reference (read-only)
  demo/NLMUpsample/NLMUpsample2.m                top-level Matlab driver
  demo/NLMUpsample/linux-mac code/cMRegularizarNLM3D_V2.c  C MEX core

minc2-simple/                MINC2 I/O submodule
  example/python/minc/io.py               load/save helpers
  example/python/resample_pytorch_nl.py   PyTorch grid_sample patterns
```

## Build Commands

Run all commands from `upsample/`:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build the C extension (run once; produces sr/_nlm_core.*.so)
python sr/_nlm_build.py

# Run all tests (interpolate tests run without C ext; others auto-skip)
pytest tests/

# Run a single test file
pytest tests/test_core.py -v

# Run tests matching a keyword
pytest tests/ -k "keyword" -v

# Run tests with coverage
pytest --cov=sr tests/

# Type checking
mypy sr/

# Linting
flake8 sr/ tests/

# Format check (CI) / auto-format (local)
black --check sr/ tests/
black sr/ tests/
```

## MINC2 Integration

The project uses `minc2-simple` for all MINC file I/O. Full API docs in `minc2-simple/USAGE.md`.

```python
from minc2_simple import minc2_file, minc2_error
from minc.io import load_minc_volume, save_minc_volume

# Load: returns (torch.Tensor, numpy.matrix voxel-to-world 4x4)
data, v2w = load_minc_volume("input.mnc")

# Save
save_minc_volume("output.mnc", tensor, v2w, ref_fname="input.mnc", history=_history)
```

Key patterns for spatial transformations with PyTorch:

```python
import torch.nn.functional as F

# Build sampling grid from affine matrix (4x4 numpy → torch)
grid = F.affine_grid(
    torch.tensor(matrix[0:3, 0:4]).unsqueeze(0).float(),
    [1, 1, *data.shape],
    align_corners=True,
)
# Resample
out = F.grid_sample(
    data.unsqueeze(0).unsqueeze(0).float(), grid,
    align_corners=True, mode="bilinear", padding_mode="zeros",
).squeeze(0).squeeze(0)
```

Always call `mm.setup_standard_order()` after opening any MINC file.
Dimension order after standard order: **Z → Y → X** (numpy C-order, slowest to fastest).
Tensor shapes in `sr/` follow **(depth, height, width)** for 3-D volumes,
or **(batch, channels, depth, height, width)** when batching for PyTorch ops.

## Code Style Guidelines

### Imports

```python
# 1. Standard library
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

# 2. Third-party
import numpy as np
import torch
import torch.nn.functional as F
from minc2_simple import minc2_file, minc2_error

```

### Formatting

- Auto-format with **Black** (line length: 88 characters)
- 4-space indentation, no tabs
- Trailing commas in multi-line containers
- Two blank lines between top-level definitions; one blank line between methods

### Naming Conventions

| Symbol | Convention | Example |
|---|---|---|
| Functions / methods | `snake_case` | `compute_weights()` |
| Variables | `snake_case` | `patch_size` |
| Classes | `PascalCase` | `SuperResolution` |
| Constants | `UPPER_SNAKE` | `DEFAULT_PATCH_SIZE` |
| Private helpers | leading `_` | `_build_grid()` |
| Type aliases | `PascalCase` | `VolumeT = torch.Tensor` |

### Type Hints

- Annotate all public function signatures.
- Use `torch.Tensor` (not `Tensor`) for clarity.
- Document tensor shapes in inline comments:

```python
def regularize(
    lr_volume: torch.Tensor,   # (D, H, W)
    prior: torch.Tensor,       # (D, H, W)
    patch_size: int = 3,
    search_radius: int = 5,
) -> torch.Tensor:             # (D, H, W)
```

- Use `Optional[X]` / `Union[X, Y]` from `typing` (Python < 3.10 compat).

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def compute_psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio between two volumes.

    Args:
        reference: Ground-truth volume, shape (D, H, W).
        estimate: Reconstructed volume, shape (D, H, W).

    Returns:
        PSNR in decibels.

    Raises:
        ValueError: If tensor shapes do not match.
    """
```

Include mathematical formulas (LaTeX notation in comments) for algorithmic functions.
Reference the paper (`Manjon 2010`) in module-level docstrings.

### Error Handling

- Raise specific built-in exceptions: `ValueError`, `RuntimeError`, `FileNotFoundError`.
- Custom exceptions inherit from `Exception` and live in `sr/exceptions.py`.
- Validate inputs at the top of each function with clear messages:

```python
if volume.ndim != 3:
    raise ValueError(f"Expected 3-D volume, got shape {tuple(volume.shape)}")
```

- Never use bare `assert` for runtime checks in library code (use `raise`).
  `assert` is acceptable in test files only.
- Catch `minc2_error` specifically when doing MINC I/O.

## Testing Guidelines

- Use **pytest** for all tests (not `unittest`).
- Test file names: `tests/test_<module>.py`.
- Use `@pytest.mark.parametrize` for multiple input scenarios.
- Mock MINC file I/O with `unittest.mock` or `pytest-mock` to avoid needing real `.mnc` files.
- Test edge cases: single-slice volumes, uniform inputs, zero-intensity masks.
- Fixtures for common tensors go in `tests/conftest.py`.

```python
import pytest
import torch

@pytest.mark.parametrize("patch_size", [3, 5, 7])
def test_regularize_shape(patch_size: int) -> None:
    vol = torch.rand(20, 20, 20)
    result = regularize(vol, vol, patch_size=patch_size)
    assert result.shape == vol.shape
```

## Performance Guidelines

- Select device once at entry point:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- Convert MINC tensors to the target device/dtype immediately after loading.
- Prefer batched tensor operations over Python loops over voxels.
- Avoid unnecessary `.cpu()` / `.numpy()` round-trips inside hot loops.
- Use `torch.no_grad()` for inference / evaluation passes.
- For large volumes, consider processing patch-by-patch to manage GPU memory.

## Logging

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Starting iteration %d, level=%d, volume shape=%s", i, level, vol.shape)
logger.debug("Convergence metric d2=%.6f (tol=%.4f)", d2, tol)
```

- Module-level logger named `__name__` (not root logger).
- Log iteration number, convergence metrics, and volume shapes at `INFO`.
- Log detailed per-voxel / per-patch stats at `DEBUG`.
- Do not configure handlers inside library modules; leave that to scripts/CLI.

## CI/CD

- Tests run on every push to `main` via pytest.
- `mypy sr/` must pass with no errors.
- `black --check sr/ tests/` and `flake8 sr/ tests/` must pass.
- Do not commit binary files or large test volumes to the repository.
