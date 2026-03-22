/*
 * NLM 3-D regularization — MEX-free C interface.
 *
 * Memory layout (matches a C-contiguous numpy array of shape (nz, ny, nx)):
 *   linear index  p = z*(ny*nx) + y*nx + x
 *   dims[0]=nx (fastest), dims[1]=ny, dims[2]=nz (slowest)
 *
 * Original algorithm: Manjon & Coupe (2010), GPL-licensed MEX source
 * cMRegularizarNLM3D_V2.c.  This header exposes the same computation via
 * a plain C function that Python/cffi can call directly.
 */

#ifndef NLM_CORE_H
#define NLM_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * nlm3d_regularize — one NLM regularization pass.
 *
 * Parameters
 * ----------
 * ima    : input volume, C-contiguous float64, shape (nz, ny, nx)
 * nx     : number of voxels along x (fastest axis, = vol.shape[2])
 * ny     : number of voxels along y  (= vol.shape[1])
 * nz     : number of voxels along z  (= vol.shape[0])
 * v      : search-neighbourhood radius (typical: 3)
 * f      : patch half-size, patch is (2f+1)^3 (typical: 1)
 * h      : per-voxel noise-level array, same shape as ima
 * fac_x  : upsampling factor along x
 * fac_y  : upsampling factor along y
 * fac_z  : upsampling factor along z
 * fima   : output buffer, caller-allocated, same size as ima
 *
 * Returns 0 on success, -1 if internal calloc fails.
 */
int nlm3d_regularize(
    const double *ima,
    int nx, int ny, int nz,
    int v, int f,
    const double *h,
    int fac_x, int fac_y, int fac_z,
    double *fima
);

#ifdef __cplusplus
}
#endif

#endif /* NLM_CORE_H */
