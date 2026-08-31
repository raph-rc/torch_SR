/*
 * ezminc_anlm_shim.cc -- in-process face of the vendored minc_anlm.
 *
 * Mirrors minc_anlm's main(): build an anlm_proc over the caller's volume and
 * run exec(1).  anlm_proc is defined in the vendored minc_anlm.cpp; its
 * declaration is reproduced here verbatim (the data members, and their order,
 * are what the constructor laid out in that file depends on) so the shim can
 * construct one and read back its fima/distances/Label volumes.
 *
 * Compiled into its own extension: the two EZminc tools cannot share one
 * ``-Dmain=`` rename.
 */

#include <string.h>
#include <vector>

#include <minc_io_simple_volume.h>

/* minc_anlm.cpp's anlm_proc, declaration reproduced verbatim. */
class anlm_proc
{
  protected:

    struct thread_parameters{
      int ini;
      int fin;
      anlm_proc* _this;
    };

  static void* worker_thread(void *arg);
  void* thread_process(int ini,int fin);

  static double bessi0(double x);
  static double bessi1(double x);
  static double Epsi(double snr);

  static void Regularize(const minc::simple_volume<double>& in,
                  minc::simple_volume<double>& out,int r);

  static double distance(const minc::simple_volume<double> & ima,
                          int x,int y,int z,
                          int nx,int ny,int nz,
                          int f);

  static double distance2(const minc::simple_volume<double> & ima,
                   const minc::simple_volume<double> & medias,
                   int x,int y,int z,
                   int nx,int ny,int nz,
                   int f);

  static void Average_block(const minc::simple_volume<double>& ima,
                     int x,int y,int z,
                     int neighborhoodsize,
                     std::vector<double> &average,
                     double weight,bool rician);

  static void Value_block(minc::simple_volume<double> &Estimate,
                   minc::simple_volume<int> &Label,
                   int x,int y,int z,
                   int neighborhoodsize,
                   const std::vector<double>  &average,
                   double global_sum);


  public:

  const minc::simple_volume<double>& ima;
  minc::simple_volume<double> fima;
  minc::simple_volume<double> means;
  minc::simple_volume<double> variances;

  minc::simple_volume<int> Label;
  minc::simple_volume<double> bias;
  minc::simple_volume<double> Estimate;
  minc::simple_volume<double> distances;

  int search_radius;
  int patch_radius;
  double imax;

  bool rician;
  double beta;
  bool debug;

  anlm_proc(const minc::simple_volume<double>&img,int search,int patch,bool rician,double beta,bool _debug=false);

  void exec(int Nthreads);
};

namespace {

size_t n_voxels(int sx, int sy, int sz)
{
  return (size_t) sx * (size_t) sy * (size_t) sz;
}

}  // namespace

extern "C" {

int ez_minc_anlm(const double *vol, int sx, int sy, int sz,
                 int search, int patch, int rician, double beta,
                 float *out_fima, double *out_distances, int *out_label)
{
  try {
    minc::simple_volume<double> src(sx, sy, sz);
    memcpy(src.c_buf(), vol, n_voxels(sx, sy, sz) * sizeof(double));

    anlm_proc anlm(src, search, patch, rician != 0, beta, false);
    anlm.exec(1);

    size_t n = n_voxels(sx, sy, sz);
    for (size_t i = 0; i < n; i++)
      out_fima[i] = (float) anlm.fima.c_buf()[i];
    if (out_distances != NULL)
      for (size_t i = 0; i < n; i++)
        out_distances[i] = anlm.distances.c_buf()[i];
    if (out_label != NULL)
      for (size_t i = 0; i < n; i++)
        out_label[i] = anlm.Label.c_buf()[i];
    return 0;
  } catch (...) {
    return -10;
  }
}

}  // extern "C"