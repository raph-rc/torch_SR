"""Thin CFFI wrappers around the vendored EZminc oracle tools.

Adapted from the EZminc-only slice of ``torch_n3/backends/legacy.py`` (this
repository has no non-uniformity-correction pipeline, so nothing else from
that module — histograms, splines, the N3 field solver — applies here, and
its N3-only ``denoise()`` refusal stub is dropped entirely).

Tensors in, tensors out. No MINC file is involved: these are the vendored
C++ routines called directly on in-memory buffers, not the installed
programs run as subprocesses.

``noise_estimate``/``dwt_forward`` are a byte-level oracle for
``sr_pytorch.noise_estimate``/``sr_pytorch.dwt`` — both are ports of the same
computation. ``mincnlm``/``minc_anlm`` use the Coupe non-local-means
convention, which is *not* the same computation as this repository's own
``sr/nlm_core.c``/``sr_pytorch/regularize.py`` (Manjon's convention) — see
``COPYING.NLM``. They are byte-verified reference implementations in their
own right, not an oracle for this project's NLM filter.

Build the extensions first::

    python3 _legacy/build_legacy.py
"""

import numpy as np
import torch

try:
    from _legacy._ezminclegacy import ffi as _ez_ffi, lib as _ez_lib
    from _legacy._ezminc_anlm import ffi as _anlm_ffi, lib as _anlm_lib
    _HAVE_LEGACY_EXT = True
except ImportError:
    _HAVE_LEGACY_EXT = False


def have_legacy_ext() -> bool:
    """Whether the EZminc CFFI extensions are built and importable."""
    return _HAVE_LEGACY_EXT


def _as_float32_array(values):
    """A contiguous float32 copy: what the EZminc tools read out of a file.

    The tools read (and write) ``float`` buffers, so the shim's volume
    pointer is ``float *``.
    """
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.ascontiguousarray(values, dtype=np.float32)


def _as_double_array(values):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.ascontiguousarray(values, dtype=np.float64)


def _as_mask_array(mask):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(mask, dtype=bool), dtype=np.uint8)


def _ez_out(array):
    return _ez_ffi.cast("float *", _ez_ffi.from_buffer(array))


def _anlm_out(array, kind):
    return _anlm_ffi.cast(kind, _anlm_ffi.from_buffer(array))


def noise_estimate(volume, mask=None, gaussian=False, hist_bins=2000):
    """Rician (or Gaussian) noise level of ``volume``, as the EZminc tool.

    Wraps the vendored ``noise_estimate.cpp``, the estimator mincnlm uses
    when its sigma is left automatic. Returns ``(noise, mean_signal)``, the
    population standard deviation of the noise and the mean of the signal it
    was estimated against. ``mask``, when given, is the object mask used
    instead of the built-in k-means detection; it is downsampled inside the
    routine exactly as the program does.

    The estimate is made on the lowest-resolution wavelet level of
    ``volume``, so the histogram has ``hist_bins`` bins there, not on the
    volume itself.
    """
    vol = _as_float32_array(volume)
    if vol.ndim != 3:
        raise ValueError("noise_estimate: volume must be 3-D (got %r)"
                         % (vol.shape,))
    sz, sy, sx = vol.shape
    mask_buf = None
    mask_ptr = _ez_ffi.NULL
    if mask is not None:
        mask_buf = _as_mask_array(mask)
        if mask_buf.shape != vol.shape:
            raise ValueError("noise_estimate: mask shape %s != volume shape %s"
                             % (mask_buf.shape, vol.shape))
        mask_ptr = _ez_ffi.cast("const unsigned char *",
                                _ez_ffi.from_buffer(mask_buf))
    out = np.zeros(2, dtype=np.float64)
    rc = _ez_lib.ez_noise_estimate(_ez_ffi.cast("const float *",
                                                _ez_ffi.from_buffer(vol)),
                                   sx, sy, sz, mask_ptr, 1 if gaussian else 0,
                                   int(hist_bins), _ez_ffi.cast("double *", _ez_ffi.from_buffer(out)),
                                   _ez_ffi.cast("double *", _ez_ffi.from_buffer(out)) + 1)
    if rc != 0:
        raise RuntimeError("noise_estimate: C++ failure (rc=%d)" % rc)
    return float(out[0]), float(out[1])


def dwt_forward(volume):
    """The one-level forward 3-D DWT ``noise_estimate`` runs its input through.

    Wraps the vendored ``dwt_forward`` (``dwt_utils.cpp`` + ``pwt`` step of
    ``dwt.cpp``): ``volume`` is zero-padded to the nearest power-of-two grid
    (centred, so a power-of-two extent stays put), transformed once in z, y,
    x, and cut back into eight subbands. Returns them as float32 tensors in
    quadrant order ``(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)``, each
    ``ceil(dim / 2)`` along every axis, x varying fastest.
    """
    vol = _as_float32_array(volume)
    if vol.ndim != 3:
        raise ValueError("dwt_forward: volume must be 3-D (got %r)"
                         % (vol.shape,))
    sz, sy, sx = vol.shape
    out = [np.zeros(((sz + 1) // 2, (sy + 1) // 2, (sx + 1) // 2),
                    dtype=np.float32) for _ in range(8)]
    rc = _ez_lib.ez_dwt_forward(
        _ez_ffi.cast("const float *", _ez_ffi.from_buffer(vol)),
        sx, sy, sz, _ez_out(out[0]), _ez_out(out[1]), _ez_out(out[2]),
        _ez_out(out[3]), _ez_out(out[4]), _ez_out(out[5]), _ez_out(out[6]),
        _ez_out(out[7]))
    if rc != 0:
        raise RuntimeError("dwt_forward: C++ failure (rc=%d)" % rc)
    return [torch.from_numpy(b) for b in out]


def mincnlm(volume, sigma=0.0, weight_method=0, M=5, S=1, block=1, b_space=2,
            m_min=0.95, v_min=0.5, beta=1.0, anisotropic=False,
            resolution=None):
    """Non-local-means denoise ``volume``, as the vendored ``mincnlm`` tool.

    ``sigma`` is the noise level (``-sigma``); 0.0 asks the program to
    estimate it from ``volume``. The remaining arguments are mincnlm's own
    options: ``weight_method`` (0 Gaussian, 1 speckle, 2 Rician), ``M``/``S``
    the search/neighbourhood sizes (``-d``/``-v``), ``block`` the blockwise
    implementation, ``b_space`` the distance between blocks, ``m_min``/
    ``v_min`` the mean/variance test bounds, ``beta`` the weight exponent,
    and ``anisotropic`` whether the neighbourhood adapts to the voxel
    resolution. ``resolution`` is the (x, y, z) voxel step used in
    anisotropic mode; it is ignored otherwise.

    Returns the denoised volume as a float32 tensor. At one thread, which is
    what the shim always uses, the result is byte-for-byte reproducible; the
    program's default of four threads is not.
    """
    vol = _as_float32_array(volume)
    if vol.ndim != 3:
        raise ValueError("mincnlm: volume must be 3-D (got %r)" % (vol.shape,))
    sz, sy, sx = vol.shape
    res_ptr = _ez_ffi.NULL
    if resolution is not None:
        res = np.ascontiguousarray(np.asarray(resolution, dtype=np.float32))
        if res.shape != (3,):
            raise ValueError("mincnlm: resolution must be (x, y, z) steps")
        res_ptr = _ez_ffi.cast("const float *", _ez_ffi.from_buffer(res))
    out = np.zeros(vol.size, dtype=np.float32)
    rc = _ez_lib.ez_mincnlm(
        _ez_ffi.cast("const float *", _ez_ffi.from_buffer(vol)),
        sx, sy, sz, res_ptr, float(sigma), int(weight_method), int(M), int(S),
        int(block), int(b_space), float(m_min), float(v_min), float(beta),
        1 if anisotropic else 0, _ez_out(out))
    if rc != 0:
        raise ValueError({
            -1: "mincnlm: weight_method=1 (speckle) needs an explicit sigma",
            -2: "mincnlm: block mode needs S >= b_space // 2",
            -3: "mincnlm: block mode is not available in anisotropic mode",
        }.get(rc, "mincnlm: C++ failure (rc=%d)" % rc))
    return torch.from_numpy(out.reshape(vol.shape))


def minc_anlm(volume, search=2, patch=1, rician=False, beta=1.0):
    """Adaptive non-local-means denoise ``volume``, as ``minc_anlm``.

    ``search``/``patch`` are the search and patch radii (``--search``,
    ``--patch``), ``rician`` corrects for Rician noise (``--rician``) and
    ``beta`` adjusts the weighting (``--beta``). Returns the denoised volume
    (its ``fima``) as a float32 tensor, matching what the tool writes with
    ``--float``.
    """
    vol = _as_double_array(volume)
    if vol.ndim != 3:
        raise ValueError("minc_anlm: volume must be 3-D (got %r)" % (vol.shape,))
    sz, sy, sx = vol.shape
    n = vol.size
    fima = np.zeros(n, dtype=np.float32)
    distances = np.zeros(n, dtype=np.float64)
    label = np.zeros(n, dtype=np.int32)
    rc = _anlm_lib.ez_minc_anlm(
        _anlm_ffi.cast("const double *", _anlm_ffi.from_buffer(vol)),
        sx, sy, sz, int(search), int(patch), 1 if rician else 0, float(beta),
        _anlm_out(fima, "float *"), _anlm_out(distances, "double *"),
        _anlm_out(label, "int *"))
    if rc != 0:
        raise RuntimeError("minc_anlm: C++ failure (rc=%d)" % rc)
    return torch.from_numpy(fima.reshape(vol.shape))
