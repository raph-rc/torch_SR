# Vendored EZminc reference tools

The EZminc programs behind the non-local-means denoisers, copied here **byte
for byte** from `torch_n3/_legacy/ezminc/` in the sibling `torch_n3` project
(itself vendored byte for byte from `legacy/EZminc/mincnlm/` and
`legacy/EZminc/image_proc/`, plus the three `minc_io_*.h` headers and
`fftw3.h` from the installed toolkit's `include/`), so that
`build_legacy.py` compiles them into CFFI extensions with nothing but GSL
and the toolkit's static FFTW. The directory names are the originals, so
`legacy/EZminc/mincnlm/<file>` maps straight onto
`upsample/_legacy/ezminc/mincnlm/<file>`.

**Do not modify these files.** They are the in-process reference for the
denoisers, compared against the *installed* programs by
`tests/test_ezminc_backend.py` through the recorded answers in
`tests/reference/legacy_ezminc.npz`. If a file needs to change to build, the
fix belongs in the shims (`ezminc_shim.cc`, `ezminc_anlm_shim.cc`) or in
`compat/`. If `torch_n3`'s own copy is re-vendored from a newer `legacy/EZminc`
checkout, re-copy this directory from it rather than editing in place, so the
two stay identical.

| File | Used for |
|---|---|
| `mincnlm/mincnlm.cpp` | mincnlm's `Exec()` and the file-scope options it reads |
| `mincnlm/minc_anlm.cpp` | `anlm_proc`, the adaptive non-local-means algorithm |
| `mincnlm/nl_means.{cpp,h}`, `nl_means_utils.{cpp,h}`, `nl_means_block.{cpp,h}` | the denoising loops mincnlm drives |
| `image_proc/noise_estimate.{cpp,h}` | the Rician/Gaussian noise estimator behind automatic sigma |
| `image_proc/minc_histograms.{cpp,h}` | the wavelet-level histogram the estimator works on |
| `image_proc/dwt.{cpp,h}`, `dwt_utils.{cpp,h}`, `fftw_blur.{cpp,h}` | the wavelet levels and low-pass blur the estimator needs |
| `image_proc/volume_dwt.{cpp,h}` | **not compiled**: the standalone `volume_dwt` tool; its `main()` does not compile against these vendored `minc_io` headers, and nothing the shim needs is defined in it — the wavelet machinery lives in `dwt.cpp`/`dwt_utils.cpp`.  Kept for reference. |
| `minc/minc_io_{simple_volume,fixed_vector,exceptions}.h`, `minc/fftw3.h` | toolkit headers the sources include; the FFTW header is vendored so no toolkit `include/` dir shadows the compat stubs below |

`compat/` supplies `minc_1_simple.h` and `minc_1_simple_rw.h`, stand-ins for
the MINC reader/writer API that the tools' `main()`s use (and the shared
`upsample/_legacy/compat/` supplies `volume_io.h`, `time_stamp.h`,
`ParseArgv.h`). Every reader/writer call lives in a `main()` the build
renames away, so the stubs abort if anything ever reaches them.

The two shims compile into **two** extensions (`_ezminclegacy`,
`_ezminc_anlm`) because each tool is a program whose `main` must be renamed
to its own unique symbol and the `-Dmain=` macro is per-build.

`mincnlm`/`minc_anlm` use the Coupe-style non-local-means convention. That is
**not** the same computation as `sr/nlm_core.c`/`sr_pytorch/regularize.py`
(Manjon's algorithm, `cMRegularizarNLM3D_V2.c`) — see `COPYING.NLM` and
`upsample/tests/test_ezminc_backend.py`'s module docstring. These extensions
are a byte-verified oracle for themselves and, through `noise_estimate`/
`dwt_forward`, for `sr_pytorch.noise_estimate`/`sr_pytorch.dwt`. They are not
an oracle for this repository's own NLM filter.

## License

EZminc carries no repository-wide license file — only per-file/per-algorithm
copyright notices. `mincnlm/minc_anlm.cpp`, for instance, carries:

```
Copyright (C) 2008 by Pierrick COUPE - L. Collins Lab
The original implementation was protected under the licence:
IDDN.FR.001.070033.000.S.P.2007.000.21000
```

Anyone redistributing this repository should settle that against the terms
of any other license applied to the rest of the codebase before doing so,
the same caveat `torch_n3/_legacy/ezminc/` is vendored under.
