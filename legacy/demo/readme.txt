/* Jose V. Manjon - jmanjon@fis.upv.es                                     */
/* Universidad Politecinca de Valencia, Spain                              */ 
/* Pierrick Coupe - pierrick.coupe@gmail.com                               */
/* Brain Imaging Center, Montreal Neurological Institute.                  */
/* Mc Gill University                                                      */
/*                                                                         */
/* Copyright (C) 2010 Jose V. Manjon and Pierrick Coupe                    */

%**************************************************************************
%                                                                         *
%              Non-Local MRI Upsampling                                   *             
%                                                                         * 
%              Jose V. Manjon, Pierrick Coupe, Antonio Buades,            *
%              Vladimir Fonov, D. Louis Collins and Montserrat Robles     *
%                                                                         *
%**************************************************************************


/*                          Details on MBONLM filter                      */
/***************************************************************************
 *  The MBONLM filter is described in:                                       *
 *                                                                         *
 *  P. Coup?, P. Yger, S. Prima, P. Hellier, C. Kervrann, C. Barillot.     *
 *  An Optimized Blockwise Non Local Means Denoising Filter for 3D Magnetic*
 *  Resonance Images. IEEE Transactions on Medical Imaging, 27(4):425-441, * 
 *  Avril 2008                                                             *
 ***************************************************************************/

/*                          Details on Wavelet mixing                     */
/***************************************************************************
 *  The hard wavelet subbands mixing is described in:                      *
 *                                                                         *
 *  P. Coup?, S. Prima, P. Hellier, C. Kervrann, C. Barillot.              *
 *  3D Wavelet Sub-Bands Mixing for Image Denoising                        *
 *  International Journal of Biomedical Imaging, 2008                      * 
 ***************************************************************************/


Matlab code for the experiments of paper:

"Non-Local MRI Upsampling"


1) Select the directory where the files have been extracted 
as work directory in matlab.

2) run the scripts:

test3D.m        (This script quantitatively compares common interpolation method with the proposed method, no noise case)     
test3Dnoisy.m   (This script quantitatively compares common interpolation method with the proposed method, noisy case)
testReal.m      (This script qualitatively compares B-spline interpolation and the proposed method over real data)

3) Go to take a coffee :)


Usage of MBONLM3D:
  
fima=MBONLM3D(nima,w,f,h,r)

nima: noisy volume
w: radius of the 3D search area (typical value, w=3)
f: radius of the 3D patch used to compute similarity (typical value, f=1)
h: noise standard deviation 
r: flag to use Gaussian or Rician adapted filtering (0:Gaussian,1:Rician)
fima: filtered volume

Usage of NLMUpsample2:
  
fima=NLMUpsample2(ima,f)

ima: LR volume
f: magnification factor in each dimension (for example [2,2,2])
fima: HR upsample volume


Note: the mex code is a multithreaded version for windows.
