"""
cffi out-of-line build script for the NLM C extension.

Run once from the ``upsample/`` project root::

    python sr/_nlm_build.py

This produces ``sr/_nlm_core.<platform>.so`` importable as ``sr._nlm_core``.
"""

import os

from cffi import FFI

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(HERE, "..")

ffi = FFI()

ffi.set_source(
    "sr._nlm_core",
    '#include "nlm_core.h"',
    sources=[os.path.join(HERE, "nlm_core.c")],
    include_dirs=[HERE],
    libraries=["m", "pthread"],
    extra_compile_args=["-O2"],
)

ffi.cdef(
    """
    int nlm3d_regularize(
        const double *ima,
        int nx, int ny, int nz,
        int v, int f,
        const double *h,
        int fac_x, int fac_y, int fac_z,
        double *fima
    );
"""
)

if __name__ == "__main__":
    ffi.compile(tmpdir=PROJECT_ROOT, verbose=True)
