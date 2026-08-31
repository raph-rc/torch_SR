"""Build the CFFI extensions that expose the vendored EZminc oracle tools.

Run directly to (re)build in place::

    python3 upsample/_legacy/build_legacy.py

Adapted from ``torch_n3/_legacy/build_legacy.py``'s ``build_ezminc()``: this
package vendors *only* the EZminc side (``noise_estimate``, ``dwt_forward``,
``mincnlm``, ``minc_anlm``) from ``ezminc/`` (see its ``README.md``), linking
GSL and the toolkit's static FFTW. It has no N3/EBTKS counterpart here —
torch_SR carries no non-uniformity-correction code, and this build script is
deliberately scoped to match.

This is a byte-verified oracle for two of the four functions it exposes:
``noise_estimate``/``dwt_forward`` match ``sr_pytorch.noise_estimate``/
``sr_pytorch.dwt`` exactly (both are ports of the same computation).
``mincnlm``/``minc_anlm`` use a different NLM convention than this
repository's own ``sr/nlm_core.c``/``sr_pytorch/regularize.py`` and are not
an oracle for those — see ``COPYING.NLM``.

``compat/`` supplies the three MINC headers the vendored sources include
(``volume_io.h``, ``time_stamp.h``, ``ParseArgv.h``), copied byte for byte
from ``torch_n3/_legacy/compat/``, so they compile unmodified without
libminc2.
"""

import os
import shutil
import sys

import cffi

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(HERE, "build")

# The EZminc reference tools, vendored byte-for-byte from torch_n3's own copy
# (itself from legacy/EZminc and the installed toolkit; see the README.md in
# ``ezminc/``). They are compiled into their *own* extensions: each tool is a
# program whose ``main`` must be renamed out of the way, and the ``-Dmain=``
# macro is per-build, so mixing mincnlm and minc_anlm under one rename would
# collide.
#
# volume_dwt.cpp is *not* compiled anywhere: it is the standalone
# ``volume_dwt`` tool, its main() does not compile against these vendored
# minc_io headers, and nothing the shim needs is defined in it -- the wavelet
# machinery it drives (dwt_forward, find_nearest_square_pow2) lives in
# dwt.cpp and dwt_utils.cpp. It is kept vendored, byte-identical, for
# reference.
EZMINC = os.path.join(HERE, "ezminc")
EZMINC_COMMON_SOURCES = [
    os.path.join(EZMINC, "mincnlm", "nl_means.cpp"),
    os.path.join(EZMINC, "mincnlm", "nl_means_block.cpp"),
    os.path.join(EZMINC, "mincnlm", "nl_means_utils.cpp"),
    os.path.join(EZMINC, "image_proc", "noise_estimate.cpp"),
    os.path.join(EZMINC, "image_proc", "minc_histograms.cpp"),
    os.path.join(EZMINC, "image_proc", "dwt.cpp"),
    os.path.join(EZMINC, "image_proc", "dwt_utils.cpp"),
    os.path.join(EZMINC, "image_proc", "fftw_blur.cpp"),
]
EZMINC_MINCNLM_SOURCES = [
    os.path.join(EZMINC, "mincnlm", "mincnlm.cpp"),
    os.path.join(EZMINC, "ezminc_shim.cc"),
] + EZMINC_COMMON_SOURCES
EZMINC_ANLM_SOURCES = [
    os.path.join(EZMINC, "mincnlm", "minc_anlm.cpp"),
    os.path.join(EZMINC, "ezminc_anlm_shim.cc"),
]
EZMINC_INCLUDE_DIRS = [
    EZMINC,
    os.path.join(EZMINC, "minc"),
    os.path.join(EZMINC, "image_proc"),
    os.path.join(EZMINC, "compat"),
    os.path.join(HERE, "compat"),   # volume_io.h / time_stamp.h / ParseArgv.h
]
#: gsl (Bessel functions) and the toolkit's static FFTW, which is PIC and so
#: links cleanly into a shared object. ``gslcblas`` rides along as libgsl's
#: dependency; it is loaded RTLD_LOCAL with this extension, so its BLAS
#: symbols never interpose on the process-wide ones PyTorch owns.
EZMINC_LIBRARIES = ["gsl", "gslcblas", "fftw3f", "m"]
EZMINC_LIBRARY_DIRS = ["/opt/minc/1.9.18.13/lib"]


def build_ezminc(verbose=True):
    """Build the two EZminc oracle extensions (``_ezminclegacy``, ``_ezminc_anlm``).

    Two separate extensions because they cannot share one ``-Dmain=`` macro:
    mincnlm's ``main`` and minc_anlm's ``main`` would collide under a single
    rename.
    """
    common = dict(
        include_dirs=EZMINC_INCLUDE_DIRS,
        library_dirs=EZMINC_LIBRARY_DIRS,
        libraries=EZMINC_LIBRARIES,
        extra_compile_args=["-O2", "-w"],
    )

    def build_one(module_name, header, sources, main_rename):
        with open(os.path.join(EZMINC, header)) as fp:
            declarations = fp.read()
        ffibuilder = cffi.FFI()
        ffibuilder.cdef(declarations)
        ffibuilder.set_source(
            module_name,
            'extern "C" {\n#include "%s"\n}\n' % header,
            source_extension=".cc",
            sources=sources,
            define_macros=[("main", main_rename)],
            **common)
        built = ffibuilder.compile(tmpdir=BUILD_DIR, verbose=verbose)
        installed = os.path.join(HERE, os.path.basename(built))
        shutil.copy2(built, installed)
        return installed

    build_one("_legacy._ezminclegacy",
              "ezminc_shim.h", EZMINC_MINCNLM_SOURCES,
              "ezminc_mincnlm_unused_main")
    build_one("_legacy._ezminc_anlm",
              "ezminc_anlm_shim.h", EZMINC_ANLM_SOURCES,
              "ezminc_anlm_unused_main")


if __name__ == "__main__":
    verbose = "-q" not in sys.argv
    build_ezminc(verbose=verbose)
