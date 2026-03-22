%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Adapative wavelet subband mixing filtering
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fima]=ANLM(ima,level)

addpath '.\denoising\wavelet'

warning off;

% The minimum of the image has to be positive due to the tests on mean and var 
mini = min(ima(:));
ima = ima + abs(mini);

%params
M=3;
alpha=1;

% Filtering with Su parameters: small patch
h=level;
fima1=MBONLM3D(ima,M,alpha,h,0);
fima1=fima1 - abs(mini);

% Filtering with So parameters: big patch 
h=level;
fima2=MBONLM3D(ima,M,alpha+1,h,0);
fima2=fima2 - abs(mini);

% Hard wavelet Coefficient Mixing
fima = hsm(fima1,fima2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%