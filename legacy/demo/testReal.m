%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Jose V. Manjon - jmanjon@fis.upv.es                                     
%   Universidad Politecinca de Valencia, Spain                               
%   Pierrick Coupe - pierrick.coupe@gmail.com                               
%   Brain Imaging Center, Montreal Neurological Institute.                  
%   Mc Gill University                                                      
%                                                                         
%   Copyright (C) 2010 Jose V. Manjon and Pierrick Coupe                    
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Test Cualitativo (visual)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath .\NLMUpsample
warning off;
clc;
clear;

% add your spm path to load analyze images 
%addpath e:\spm8 

% read LR data
V1=spm_vol('T2LR.img');
ima=spm_read_vols(V1);
ima=double(ima);

% small volume (comment this line to process the whole volume)
%ima=ima(100:150,100:150,:);

% interpolation factors
lf=[1,1,3];

% B-Spline interpolation
bima=BSplineInterpolation(ima,lf);

% Proposed mentod
tic
[lima]=NLMUpsample2(ima,lf);
toc

% Write results

V1.fname='splineT2.img';
V1.dim=size(bima);
V1.mat(3,3)=V1.mat(3,3)/3;
spm_write_vol(V1,bima);

V1.fname='newT2.img';
spm_write_vol(V1,lima);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%