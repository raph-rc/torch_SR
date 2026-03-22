/*
 * NLM 3-D regularization — MEX-free implementation.
 *
 * Derived from cMRegularizarNLM3D_V2.c
 *   Copyright (C) 2010 Jose V. Manjon and Pierrick Coupe
 *
 * Changes vs. the original MEX source:
 *   - Removed mex.h / matrix.h; replaced Matlab allocators with calloc/free.
 *   - mexFunction() replaced by nlm3d_regularize() with a plain C signature.
 *   - Unused variables (salir, retval, indice, ndim) removed.
 *   - in_image stored as double* in the thread struct (const-cast at setup).
 *
 * Memory layout: C-contiguous, shape (nz, ny, nx).
 *   index  p = z*(ny*nx) + y*nx + x
 *   In the code below: dims[0]=nx, dims[1]=ny, dims[2]=nz,
 *   variable i→x, j→y, k→z, rc = nx*ny.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "nlm_core.h"

/* ------------------------------------------------------------------ */
/* Thread argument block                                               */
/* ------------------------------------------------------------------ */
struct myargument
{
    int rows;           /* ny */
    int cols;           /* nx */
    int slices;         /* nz */
    double *in_image;   /* read-only inside the thread */
    double *out_image;
    double *mean_image;
    double *pesos;
    int ini;
    int fin;
    int radio;          /* search radius v */
    int f;              /* patch radius   */
    double *sigma;      /* per-voxel noise level h */
};

/* ------------------------------------------------------------------ */
/* Patch distance (L2 in (2f+1)^3 neighbourhood, mirror BC)           */
/* ------------------------------------------------------------------ */
static double distancia(
    double *ima,
    int x, int y, int z,
    int nx, int ny, int nz,
    int f,
    int sx, int sy, int sz)
{
    double distancetotal = 0.0;
    int i, j, k, ni1, nj1, ni2, nj2, nk1, nk2;

    for (k = -f; k <= f; k++)
    {
        nk1 = z + k;
        nk2 = nz + k;
        if (nk1 < 0)  nk1 = -nk1;
        if (nk2 < 0)  nk2 = -nk2;
        if (nk1 >= sz) nk1 = 2*sz - nk1 - 1;
        if (nk2 >= sz) nk2 = 2*sz - nk2 - 1;

        for (j = -f; j <= f; j++)
        {
            nj1 = y + j;
            nj2 = ny + j;
            if (nj1 < 0)  nj1 = -nj1;
            if (nj2 < 0)  nj2 = -nj2;
            if (nj1 >= sy) nj1 = 2*sy - nj1 - 1;
            if (nj2 >= sy) nj2 = 2*sy - nj2 - 1;

            for (i = -f; i <= f; i++)
            {
                ni1 = x + i;
                ni2 = nx + i;
                if (ni1 < 0)  ni1 = -ni1;
                if (ni2 < 0)  ni2 = -ni2;
                if (ni1 >= sx) ni1 = 2*sx - ni1 - 1;
                if (ni2 >= sx) ni2 = 2*sx - ni2 - 1;

                {
                    double diff =
                        ima[nk1*(sx*sy) + nj1*sx + ni1] -
                        ima[nk2*(sx*sy) + nj2*sx + ni2];
                    distancetotal += diff * diff;
                }
            }
        }
    }

    return distancetotal / (double)((2*f+1)*(2*f+1)*(2*f+1));
}

/* ------------------------------------------------------------------ */
/* Per-thread NLM filter                                               */
/* ------------------------------------------------------------------ */
static void *ThreadFunc(void *pArguments)
{
    double *ima, *fima, *medias, *pesos, w, d, *h, t1;
    int ii, jj, kk, ni, nj, nk, i, j, k, ini, fin;
    int rows, cols, slices, v, p, p1, f, rc;

    struct myargument arg = *(struct myargument *)pArguments;

    rows   = arg.rows;
    cols   = arg.cols;
    slices = arg.slices;
    ini    = arg.ini;
    fin    = arg.fin;
    ima    = arg.in_image;
    fima   = arg.out_image;
    medias = arg.mean_image;
    pesos  = arg.pesos;
    v      = arg.radio;
    f      = arg.f;
    h      = arg.sigma;
    rc     = rows * cols;

    for (k = ini; k < fin; k++)
    {
        for (j = 0; j < rows; j++)
        {
            for (i = 0; i < cols; i++)
            {
                p = k*rc + j*cols + i;
                if (h[p] < 1) continue;

                for (kk = 0; kk <= v; kk++)
                {
                    nk = k + kk;
                    for (ii = -v; ii <= v; ii++)
                    {
                        ni = i + ii;
                        for (jj = -v; jj <= v; jj++)
                        {
                            nj = j + jj;

                            /* upper-triangle only; symmetric update below */
                            if (kk == 0 && jj < 0)        continue;
                            if (kk == 0 && jj == 0 && ii <= 0) continue;

                            if (ni >= 0 && nj >= 0 && nk >= 0 &&
                                ni < cols && nj < rows && nk < slices)
                            {
                                p1 = nk*rc + nj*cols + ni;

                                t1 = fabs(medias[p] - medias[p1]);
                                if (t1 > 0.6 * h[p]) continue;

                                d = distancia(ima,
                                              i, j, k,
                                              ni, nj, nk,
                                              f,
                                              cols, rows, slices);

                                if (h[p] > 0) d = d / (2.0*h[p]*h[p]) - 1.0;
                                if (d < 0)    d = 0.0;

                                w = exp(-d);

                                /* symmetric accumulation (race-safe
                                   approximation, same as original) */
                                fima[p]  += w * ima[p1];
                                pesos[p] += w;
                                fima[p1] += w * ima[p];
                                pesos[p1]+= w;
                            }
                        }
                    }
                }
            }
        }
    }

    pthread_exit(0);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */
int nlm3d_regularize(
    const double *ima,
    int nx, int ny, int nz,
    int v, int f,
    const double *h,
    int fac_x, int fac_y, int fac_z,
    double *fima)
{
    double *medias, *pesos, *tmp;
    double off, media;
    int i, j, k, ii, jj, kk, ni, nj, nk, Nthreads, rc, ft;
    int dims[3];   /* dims[0]=nx, dims[1]=ny, dims[2]=nz */
    int fac[3];    /* fac[0]=fac_x, fac[1]=fac_y, fac[2]=fac_z */
    int total;

    struct myargument *ThreadArgs = NULL;
    pthread_t         *ThreadList = NULL;

    dims[0] = nx;  dims[1] = ny;  dims[2] = nz;
    fac[0]  = fac_x; fac[1] = fac_y; fac[2] = fac_z;
    total   = nx * ny * nz;
    rc      = dims[0] * dims[1];   /* nx * ny */

    /* Allocate internal work buffers */
    medias = (double *)calloc(total, sizeof(double));
    tmp    = (double *)calloc(total, sizeof(double));
    pesos  = (double *)calloc(total, sizeof(double));

    if (!medias || !tmp || !pesos)
    {
        free(medias); free(tmp); free(pesos);
        return -1;
    }

    /* ---- Compute 3×3×3 local means (mirror BC) ---- */
    for (k = 0; k < dims[2]; k++)
    {
        for (j = 0; j < dims[1]; j++)
        {
            for (i = 0; i < dims[0]; i++)
            {
                media = 0.0;
                for (ii = -1; ii <= 1; ii++)
                {
                    ni = i + ii;
                    if (ni < 0)         ni = -ni;
                    if (ni >= dims[0])  ni = 2*dims[0] - ni - 1;
                    for (jj = -1; jj <= 1; jj++)
                    {
                        nj = j + jj;
                        if (nj < 0)         nj = -nj;
                        if (nj >= dims[1])  nj = 2*dims[1] - nj - 1;
                        for (kk = -1; kk <= 1; kk++)
                        {
                            nk = k + kk;
                            if (nk < 0)         nk = -nk;
                            if (nk >= dims[2])  nk = 2*dims[2] - nk - 1;

                            media += ima[nk*rc + nj*dims[0] + ni];
                        }
                    }
                }
                medias[k*rc + j*dims[0] + i] = media / 27.0;
            }
        }
    }

    /* ---- Block-wise downsampled means (for mean-preservation) ---- */
    ft = fac[0] * fac[1] * fac[2];
    for (k = 0; k < dims[2]/fac[2]; k++)
        for (j = 0; j < dims[1]/fac[1]; j++)
            for (i = 0; i < dims[0]/fac[0]; i++)
            {
                media = 0.0;
                for (kk = 0; kk < fac[2]; kk++)
                    for (jj = 0; jj < fac[1]; jj++)
                        for (ii = 0; ii < fac[0]; ii++)
                            media += ima[(k*fac[2]+kk)*rc +
                                         (j*fac[1]+jj)*dims[0] +
                                          i*fac[0]+ii];
                tmp[k*rc + j*dims[0] + i] = media / (double)ft;
            }

    /* ---- Initialise output: fima = ima, pesos = 1 ---- */
    for (i = 0; i < total; i++)
    {
        fima[i]  = ima[i];
        pesos[i] = 1.0;
    }

    /* ---- Thread dispatch ---- */
    Nthreads = (dims[2] < 8) ? 1 : dims[2] / 8;

    ThreadList = (pthread_t *)        calloc(Nthreads, sizeof(pthread_t));
    ThreadArgs = (struct myargument *) calloc(Nthreads, sizeof(struct myargument));

    if (!ThreadList || !ThreadArgs)
    {
        free(medias); free(tmp); free(pesos);
        free(ThreadList); free(ThreadArgs);
        return -1;
    }

    for (i = 0; i < Nthreads; i++)
    {
        ThreadArgs[i].ini        = (int)((i   * dims[2]) / Nthreads);
        ThreadArgs[i].fin        = (int)(((i+1)* dims[2]) / Nthreads);
        ThreadArgs[i].cols       = dims[0];
        ThreadArgs[i].rows       = dims[1];
        ThreadArgs[i].slices     = dims[2];
        ThreadArgs[i].in_image   = (double *)ima;  /* thread only reads */
        ThreadArgs[i].out_image  = fima;
        ThreadArgs[i].mean_image = medias;
        ThreadArgs[i].pesos      = pesos;
        ThreadArgs[i].radio      = v;
        ThreadArgs[i].f          = f;
        ThreadArgs[i].sigma      = (double *)h;    /* thread only reads */
    }

    for (i = 0; i < Nthreads; i++)
    {
        if (pthread_create(&ThreadList[i], NULL, ThreadFunc, &ThreadArgs[i]))
        {
            fprintf(stderr, "nlm3d_regularize: pthread_create failed\n");
            free(medias); free(tmp); free(pesos);
            free(ThreadList); free(ThreadArgs);
            return -1;
        }
    }

    for (i = 0; i < Nthreads; i++)
        pthread_join(ThreadList[i], NULL);

    free(ThreadArgs);
    free(ThreadList);

    /* ---- Normalise by accumulated weights ---- */
    for (i = 0; i < total; i++)
        fima[i] /= pesos[i];

    /* ---- Mean-preservation constraint ---- */
    for (k = 0; k < dims[2]; k += fac[2])
        for (j = 0; j < dims[1]; j += fac[1])
            for (i = 0; i < dims[0]; i += fac[0])
            {
                media = 0.0;
                for (kk = 0; kk < fac[2]; kk++)
                    for (jj = 0; jj < fac[1]; jj++)
                        for (ii = 0; ii < fac[0]; ii++)
                            media += fima[(k+kk)*rc + (j+jj)*dims[0] + i+ii];
                media /= (double)ft;

                off = tmp[(k/fac[2])*rc + (j/fac[1])*dims[0] + (i/fac[0])] - media;

                for (kk = 0; kk < fac[2]; kk++)
                    for (jj = 0; jj < fac[1]; jj++)
                        for (ii = 0; ii < fac[0]; ii++)
                            fima[(k+kk)*rc + (j+jj)*dims[0] + i+ii] += off;
            }

    free(medias);
    free(tmp);
    free(pesos);
    return 0;
}
