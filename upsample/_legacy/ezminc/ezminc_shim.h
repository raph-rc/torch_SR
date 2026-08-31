/*
 * ezminc_shim.h -- C ABI over the vendored EZminc mincnlm / noise_estimate.
 *
 * Compiled as C++ *and* fed verbatim to cffi's cdef(), so it must stay free of
 * preprocessor directives and C++ constructs.
 *
 * Each function is the array-in / array-out face of one of the EZminc tools:
 * the same C++ routines the installed programs call, driven in-process on
 * caller-owned buffers instead of MINC files.  Volumes are C-ordered with
 * x varying fastest (storage order of the MINC files the tools read).
 *
 * minc_anlm lives in its own extension (ezminc_anlm_shim.h); the two tools
 * cannot share one ``-Dmain=`` rename.
 *
 * Return convention: 0 on success, negative on failure.
 */

/* mincnlm + noise_estimate (legacy/EZminc/mincnlm, image_proc):
 * automatic non-local-means denoising, and the Rician/Gaussian noise
 * estimator behind its automatic sigma.
 *
 * mask, gaussian, hist_bins drive the noise_estimate call; the rest are
 * mincnlm's own options.  res (may be NULL) is the voxel resolution, used
 * only in anisotropic mode.  Writes {noise, mean_signal} and the denoised
 * volume.  Errors: -1 speckle needs explicit sigma, -2 block needs
 * S >= b_space/2, -3 block is not available in anisotropic mode. */
int ez_noise_estimate(const float *vol, int sx, int sy, int sz,
                      const unsigned char *mask, int gaussian, int hist_bins,
                      double *noise, double *mean_signal);

int ez_mincnlm(const float *vol, int sx, int sy, int sz,
               const float *res, double sigma, int weight_method,
               int m, int s, int block, int b_space,
               double m_min, double v_min, double beta, int anisotropic,
               float *out);

/* The one-level 3-D forward DWT (minc::dwt_forward, pwt step) behind
 * noise_estimate, exposed for the port's block tests: the 8 subbands it
 * writes, in quadrant order j=0..7, i.e. LLL, LLH, LHL, LHH, HLL, HLH,
 * HHL, HHH.  Each out buffer holds ceil(sz/2)*ceil(sy/2)*ceil(sx/2)
 * floats, x varying fastest, same storage order as the input. */
int ez_dwt_forward(const float *vol, int sx, int sy, int sz,
                   float *out_lll, float *out_llh, float *out_lhl,
                   float *out_lhh, float *out_hll, float *out_hlh,
                   float *out_hhl, float *out_hhh);