# Research Notes: NLM MRI Upsampling and MINC2 File Format

Reference implementation: Manjon JV, Coupe P, Buades A, Collins DL, Robles M.
*MRI superresolution using self-similarity and image priors.*
Int J Biomed Imaging. 2010.

---

## Directory Layout

```
/app/
├── legacy/demo/NLMUpsample/      ← original Matlab + MEX C source
├── minc2-simple/example/python/  ← MINC2 Python bindings (minc2_simple)
├── upsample/                     ← Python re-implementation
│   ├── sr/                       ← core library package
│   │   ├── nlm_core.h/.c         ← MEX-free C kernel
│   │   ├── _nlm_build.py         ← CFFI build script
│   │   ├── regularize.py         ← Python wrapper around C kernel
│   │   ├── interpolate.py        ← initial B-spline interpolation
│   │   ├── core.py               ← main pipeline (nlm_upsample)
│   │   └── metrics.py            ← PSNR metric
│   ├── tests/                    ← pytest test suite
│   ├── scripts/                  ← utilities (generate_test_blocks.py)
│   ├── data/                     ← generated .npz regression fixtures
│   └── upsample.py               ← CLI entry point
└── VF_20190417_MP2RAGE.mnc       ← HR test volume (260×228×192)
```

---

## Algorithm Overview

The algorithm iteratively refines a spline-interpolated initial estimate using
Non-Local Means (NLM) regularization.  Each NLM step replaces every voxel with a
weighted average of its search neighbourhood; weights are computed from the
exponential of the negated normalised patch-distance.

### Five-stage pipeline (`nlm_upsample`)

1. **Normalise** input to [0, 256]  (`ima * 256 / max`)
2. **Initial interpolation** — cubic B-spline with mean correction
3. **Noise level estimation** — local std → smoothed → halved
4. **Iterative NLM regularization** — two-phase convergence with level halving
5. **Rescale** back to original intensity range

---

## Initial Interpolation (`sr/interpolate.py`)

### Coordinate convention

LR voxel at integer index `iz` is placed at HR coordinate:

```
hr_coord = iz * fz + (fz - 1) / 2
```

This centres each LR sample in its corresponding HR block.  In the Matlab code
this appears as `ori = (1 + lf) / 2` (1-based indexing).

### Steps

1. Build a `RegularGridInterpolator(method="cubic")` on the LR grid using the
   shifted sample positions above.
2. Evaluate at every HR grid point.
3. **Boundary replication**: edge slices outside the extrapolation range are
   repeated from the nearest valid slice.
4. **Mean preservation**: for every HR block of size `(fz, fy, fx)`, add a
   constant offset so that `mean(block) == original_LR_voxel`.  This is the
   `InitialInterpolation` constraint from the paper.

---

## NLM Regularization — C Kernel (`sr/nlm_core.c`)

Derived from `legacy/demo/NLMUpsample/linux-mac code/cMRegularizarNLM3D_V2.c`.

### C function signature

```c
int nlm3d_regularize(
    const double *ima,            // input volume
    int nx, int ny, int nz,       // C dims: nx=fastest, nz=slowest
    int v, int f,                 // search radius, patch half-size
    const double *h,              // per-voxel noise level (same layout as ima)
    int fac_x, int fac_y, int fac_z,  // upsampling factors (MINC x/y/z order)
    double *fima                  // output buffer, caller-allocated
);   // returns 0 on success, -1 on alloc failure
```

### Memory layout

C-contiguous (Z-Y-X):

```
linear index = z * (ny * nx) + y * nx + x
```

Python ↔ C parameter mapping:

| Python ndarray axis | numpy shape index | C parameter |
|---------------------|-------------------|-------------|
| Z (depth, slowest)  | `shape[0]`        | `nz`        |
| Y (height)          | `shape[1]`        | `ny`        |
| X (width, fastest)  | `shape[2]`        | `nx`        |

No transposition is needed — the MEX code uses the same row-major convention
(`p = k*(ny*nx) + j*nx + i` with `dims[0]=nx` as fastest index).

### Algorithm steps

1. Compute 3×3×3 local means (`medias` array) — used for fast pre-rejection.
2. Compute block-averaged means (`tmp`) — used for mean-preservation post-correction.
3. Initialise output `fima[i] = ima[i]`, weights `pesos[i] = 1.0`.
4. Dispatch `nthreads = (nz < 8) ? 1 : nz / 8` pthreads, each responsible for
   a contiguous range of Z-slices.
5. **Per-thread inner loop** over every voxel `(x, y, z)` with `h[p] >= 1.0`:
   - Iterate over the *upper triangle* of the `(2v+1)^3` search neighbourhood
     (symmetry avoids double work).
   - **Mean-difference filter**: skip pair if
     `|medias[p] - medias[p1]| > 0.6 * h[p]`.
   - **Patch distance** (L2 over `(2f+1)^3`, mirror BC):
     `d = Σ(patch_diff²) / (2f+1)^3`
   - **NLM weight**: `w = exp(-(d/(2·h²) - 1))`,  clamped so exponent ≤ 0.
   - **Symmetric accumulation**: `fima[p] += w*ima[p1]`,
     `fima[p1] += w*ima[p]` (and matching weight accumulators).
     Race conditions are accepted as an approximation (matches original).
6. Normalise: `fima[i] /= pesos[i]`.
7. **Mean-preservation correction**: for each `(fac_z × fac_y × fac_x)` block,
   shift all voxels so that the block mean matches `tmp`.

### Boundary conditions

Mirror reflection for all out-of-bounds accesses:

```c
if (idx < 0)     idx = -idx;
if (idx >= size) idx = 2*size - idx - 1;
```

---

## Noise Level Estimation (`sr/core.py`)

Matches `stdfilt(bima, ones(3,3,3))` → `padarray/convn` → `/2` in Matlab:

```python
sigma    = sqrt(max(E[x²] - (E[x])², 0))   # uniform_filter, size=3
smoothed = uniform_filter(sigma, size=3, mode='reflect')
level    = smoothed / 2.0
```

---

## Convergence (`sr/core.py`)

Two-phase multi-scale convergence:

```
d[i]  = mean |F - lima|   (per iteration)
ds[j] = mean |last - lima| (at each noise-level halving)

Phase 1: continue until d[i-1] / d[i] < tol (1.2)
  → halve level, record ds
  → if len(ds) >= 2 and ds[-2]/ds[-1] < tol → STOP

Absolute fallback: stop when d[i] <= 0.001
Hard limit: stop when ii+1 >= max_iter (if max_iter > 0)
```

---

## CFFI Build (`sr/_nlm_build.py`)

Out-of-line API build:

```python
ffi.set_source(
    "sr._nlm_core",
    '#include "nlm_core.h"',
    sources=["nlm_core.c"],
    include_dirs=[HERE],
    libraries=["m", "pthread"],
    extra_compile_args=["-O2"],
)
ffi.cdef("int nlm3d_regularize(const double*, int, int, int,"
         " int, int, const double*, int, int, int, double*);")
```

Run once: `python upsample/sr/_nlm_build.py`
Output: `sr/_nlm_core.<platform-tag>.so`

`have_c_ext()` in `sr/regularize.py` probes for this `.so`; returns `False`
if absent (tests skip, CLI aborts).

---

## PSNR Metric (`sr/metrics.py`)

Matches Matlab `test3D.m` (`umbral = 10`):

```
R    = reference.max() - reference.min()
mask = reference > 10.0
RMSE = sqrt(mean((reference[mask] - estimate[mask])²))
PSNR = 20 · log10(R / RMSE)
```

Returns `0.0` if no foreground voxels; `inf` if RMSE == 0.

---

## MINC2 Python API (`minc2-simple/example/python/`)

### High-level helpers (`minc/io.py`)

```python
data, v2w = load_minc_volume(fname)          # → (ndarray, 4×4 matrix)
save_minc_volume(fn, data, v2w,
                 ref_fname=None, history=None)
```

`v2w` is a 4×4 `numpy.matrix` mapping voxel index (i,j,k,1) to world (x,y,z,1).

### Low-level (`minc2_simple.minc2_file`)

```python
m = minc2_file(path)
m.setup_standard_order()   # MUST call after open; reorders to T→Z→Y→X, positive steps
data = m.load_complete_volume(MINC2_DOUBLE)
m.close()                  # MUST call before re-reading
```

### Affine decomposition (`minc/geo.py`)

```python
start, step, dir_cos = decompose(v2w)
# - start: world coordinates of first voxel (x, y, z)
# - step:  voxel spacing in each dimension (can be negative)
# - dir_cos: 3×3 orthonormal rotation matrix

v2w = compose(start, step, dir_cos)
# Reconstruction:
#   v2w[0:3, 0:3] = dir_cos @ diag(step)
#   v2w[0:3, 3]   = dir_cos @ start
```

SVD-based decomposition: `R = U @ Vh` from `svd(v2w[0:3, 0:3])`.

### MINC dimension order vs numpy axis order

| MINC dimension index | Axis label | numpy axis after `setup_standard_order` |
|----------------------|------------|-----------------------------------------|
| 0 (step[0])          | X          | 2 (fastest, width)                      |
| 1 (step[1])          | Y          | 1 (height)                              |
| 2 (step[2])          | Z          | 0 (slowest, depth)                      |

### Geometry update for upsampled output (`upsample.py`)

```python
def _adjusted_v2w(v2w, factors):        # factors = (fz, fy, fx)
    start, step, dir_cos = decompose(v2w)
    minc_factors = [fx, fy, fz]         # MINC x/y/z order
    new_step  = step / minc_factors
    new_start = start - (minc_factors - 1) / 2.0 * new_step
    return compose(new_start, new_step, dir_cos)
```

This places LR voxel `k` at HR position `k·f + (f-1)/2`, consistent with
`initial_interpolation`.

---

## CLI (`upsample/upsample.py`)

```bash
python upsample/upsample.py input.mnc output.mnc \
    --factors fz fy fx   # default: 2 1 1
    --v 3                # search radius
    --f 1                # patch half-size
    --tol 1.2            # convergence threshold
    --max-iter N         # 0 = converge automatically
    --initial-only       # spline only, skip NLM
    --verbose
```

Path bootstrap prepends `upsample/` and `minc2-simple/example/python/` to
`sys.path` so `sr` and `minc` are importable regardless of CWD.

---

## Test Suite

### Test files

| File | Requires C ext | Tests |
|------|---------------|-------|
| `test_interpolate.py` | No | shape, mean-preservation, NaN/Inf, identity (factor=1), validation |
| `test_regularize.py`  | Yes (skip) | shape, NaN/Inf, sigma<1 → identity, validation |
| `test_core.py`        | Yes (skip) | isotropic/anisotropic shape, intensity range, uniform input, validation |
| `test_psnr.py`        | Yes (skip) | NLM > spline, NLM PSNR floor, spline PSNR floor |

### PSNR regression fixtures

Generated once by `scripts/generate_test_blocks.py` from
`VF_20190417_MP2RAGE.mnc` (260×228×192 MP2RAGE brain volume):

| factor | LR side | HR side | approx size |
|--------|---------|---------|-------------|
| 2      | 12      | 24      | ~103 KB     |
| 3      | 12      | 36      | ~313 KB     |
| 5      | 10      | 50      | ~808 KB     |
| 7      | 8       | 56      | ~1115 KB    |

Brain centre voxel: `(CZ=151, CY=99, CX=95)` (median of voxels > 500).
Downsampling: `lr = hr.reshape(N,f,N,f,N,f).mean(axis=(1,3,5))`.

### Calibrated PSNR floors (measured at `max_iter=3, v=3, f=1`)

| factor | spline PSNR | NLM PSNR | spline floor | NLM floor |
|--------|-------------|----------|--------------|-----------|
| 2      | 25.615 dB   | 27.158 dB | 23.6 dB      | 26.1 dB   |
| 3      | 21.919 dB   | 22.462 dB | 19.9 dB      | 21.4 dB   |
| 5      | 18.823 dB   | 19.034 dB | 16.8 dB      | 18.0 dB   |
| 7      | 17.227 dB   | 17.316 dB | 15.2 dB      | 16.3 dB   |

Floors: NLM observed − 1.0 dB; spline observed − 2.0 dB.

---

## Key Numerical Behaviours

| Behaviour | Value / Mechanism |
|-----------|-------------------|
| Input normalisation | [0, 256] during all C/Python processing |
| Skip low-noise voxels | `h[p] < 1.0` → no NLM weighting, voxel returned unchanged |
| Mean-difference pre-filter | skip pair if `|mean_p − mean_p1| > 0.6·h[p]` |
| Weight formula | `w = exp(−max(d/(2h²) − 1, 0))` |
| Convergence guard | `1e-12` added to denominator in ratio test |
| Race-condition policy | accepted; matches original Matlab/MEX behaviour |
| Reflection boundary | all C and scipy calls use mirror/reflect mode |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | array operations |
| `scipy.ndimage` | uniform_filter (smoothing) |
| `scipy.interpolate` | RegularGridInterpolator (cubic spline) |
| `cffi` | C extension interface |
| `minc2_simple` | MINC2 file I/O (optional, CLI only) |
| `torch` | optional; auto-converted to numpy if minc2_simple returns tensors |

Build: GCC/Clang, `libm`, `libpthread`, Python dev headers.

---

## Quick-start Commands

```bash
# Build C extension (run once)
python upsample/sr/_nlm_build.py

# Generate PSNR test fixtures (run once, needs VF_20190417_MP2RAGE.mnc)
python upsample/scripts/generate_test_blocks.py

# Run full test suite
cd /app/upsample && python -m pytest tests/ -v

# Upsample a MINC2 volume (Z-only, factor 2)
python upsample/upsample.py VF_20190417_MP2RAGE_lr.mnc out.mnc --factors 2 1 1
```
