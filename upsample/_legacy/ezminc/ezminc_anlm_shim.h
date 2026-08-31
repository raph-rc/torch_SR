/*
 * ezminc_anlm_shim.h -- C ABI over the vendored EZminc minc_anlm.
 *
 * Compiled as C++ *and* fed verbatim to cffi's cdef(), so it must stay free of
 * preprocessor directives and C++ constructs.
 *
 * Volumes are C-ordered with x varying fastest.  Return convention: 0 on
 * success, negative on failure.
 */
int ez_minc_anlm(const double *vol, int sx, int sy, int sz,
                 int search, int patch, int rician, double beta,
                 float *out_fima, double *out_distances, int *out_label);