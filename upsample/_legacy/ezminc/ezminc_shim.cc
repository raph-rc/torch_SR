/*
 * ezminc_shim.cc -- in-process face of the vendored mincnlm / noise_estimate.
 *
 * Mirrors what each tool's main() does, but on caller-owned arrays instead of
 * MINC files: the shim reads its volume into the same simple_volume<float>
 * buffer the file reader would have filled, sets the same file-scope globals
 * mincnlm's argument parser sets, and calls the same routines.
 *
 * minc_anlm's shim is a separate extension (ezminc_anlm_shim.cc).
 *
 * Nothing here is reached from a renamed-away main(): the compat headers see
 * to that.  Errors come back as negative return codes, never across the C ABI.
 */

#include <string.h>

#include <volume_io.h>             /* compat: VIO_Real */
#include <minc_io_simple_volume.h>
#include "dwt_utils.h"
#include "noise_estimate.h"

/* mincnlm.cpp's file-scope globals, exactly as its argument table would set
 * them.  `testmean`/`testvar`/`nb_thread`/`block` are already declared extern
 * by the vendored nl_means.h; the rest live only in mincnlm.cpp, so they are
 * redeclared here to be settable. */
extern int verbose, clobber;
extern double filtering_param, beta_param;
extern int anisotropic;
extern double S, M;
extern int neighborhoodsize[3], searching[3];
extern int testmean, testvar, nb_thread;
extern double m_min, v_min;
extern int weight_method, block, b_space, debug, references;
extern char *hallucinate_file;

void Exec(minc::simple_volume<float> &in, float *ima_out, int *vol_size,
          VIO_Real *vol_res, float *hallucinate_in);

namespace {

size_t n_voxels(int sx, int sy, int sz)
{
  return (size_t) sx * (size_t) sy * (size_t) sz;
}

}  // namespace

extern "C" {

int ez_noise_estimate(const float *vol, int sx, int sy, int sz,
                      const unsigned char *mask, int gaussian, int hist_bins,
                      double *noise, double *mean_signal)
{
  try {
    minc::simple_volume<float> input(sx, sy, sz);
    memcpy(input.c_buf(), vol, n_voxels(sx, sy, sz) * sizeof(float));

    minc::simple_volume<unsigned char> mask_vol;
    if (mask != NULL) {
      mask_vol.resize(sx, sy, sz);
      memcpy(mask_vol.c_buf(), mask, n_voxels(sx, sy, sz));
    }

    double mean_val = 0.0;
    *noise = minc::noise_estimate(input, mean_val, gaussian != 0, false,
                                  hist_bins, mask_vol);
    *mean_signal = mean_val;
    return 0;
  } catch (...) {
    return -10;
  }
}

int ez_dwt_forward(const float *vol, int sx, int sy, int sz,
                   float *out_lll, float *out_llh, float *out_lhl,
                   float *out_lhh, float *out_hll, float *out_hlh,
                   float *out_hhl, float *out_hhh)
{
  try {
    minc::simple_volume<float> src(sx, sy, sz);
    memcpy(src.c_buf(), vol, n_voxels(sx, sy, sz) * sizeof(float));

    std::vector<minc::simple_volume<float> > dst;
    minc::dwt_forward(src, dst);

    float *outs[8] = {out_lll, out_llh, out_lhl, out_lhh,
                      out_hll, out_hlh, out_hhl, out_hhh};
    for (int j = 0; j < 8; j++)
      memcpy(outs[j], dst[j].c_buf(),
             (size_t) dst[j].dim(0) * dst[j].dim(1) * dst[j].dim(2)
                 * sizeof(float));
    return 0;
  } catch (...) {
    return -10;
  }
}

int ez_mincnlm(const float *vol, int sx, int sy, int sz,
               const float *res, double sigma, int weight_method,
               int m, int s, int block, int b_space,
               double m_min, double v_min, double beta, int anisotropic,
               float *out)
{
  /* Exec's own failure modes, checked here so they cannot reach its exit()s. */
  if (weight_method == 1 && sigma == 0.0)
    return -1;
  if (block == 1 && (int) s < b_space / 2)
    return -2;
  if (block == 1 && anisotropic)
    return -3;

  try {
    filtering_param = sigma;
    beta_param = beta;
    ::anisotropic = anisotropic;
    ::S = s;
    ::M = m;
    ::weight_method = weight_method;
    ::block = block;
    ::b_space = b_space;
    ::m_min = m_min;
    ::v_min = v_min;
    ::testmean = 1;
    ::testvar = 1;
    nb_thread = 1;
    debug = 0;
    references = 0;
    verbose = 0;
    clobber = 0;
    hallucinate_file = NULL;

    minc::simple_volume<float> in(sx, sy, sz);
    memcpy(in.c_buf(), vol, n_voxels(sx, sy, sz) * sizeof(float));

    int vol_size[3] = {sx, sy, sz};
    VIO_Real vol_res[3];
    if (res != NULL) {
      vol_res[0] = res[0];
      vol_res[1] = res[1];
      vol_res[2] = res[2];
    } else {
      vol_res[0] = vol_res[1] = vol_res[2] = 1.0;
    }

    Exec(in, out, vol_size, vol_res, NULL);
    return 0;
  } catch (...) {
    return -10;
  }
}

}  // extern "C"