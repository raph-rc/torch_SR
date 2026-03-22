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
%**************************************************************************
%                                                                         *
%    This script quantitatively compares common interpolation method      *
%    with the proposed method, no noise case.                             *
%                                                                         *
%**************************************************************************

addpath .\NLMUpsample
addpath .\denoising
warning off;
clc;
clear;

% read T1 volume
name ='t1_icbm_normal_1mm_pn0_rf0.rawb';
fid = fopen(name,'r');    
s=[181,217,181];
ima1=zeros(s(1:3));
for z=1:s(3),    
  ima1(:,:,z) = fread(fid,s(1:2),'uchar');
end;
fclose(fid);

ima1=double(ima1(1:180,1:216,1:180));

% small volume (comment this line to process the whole volume)
ima1=ima1(101:150,101:150,101:150);

% reduce ima2 un factor f
s=size(ima1);
umbral=10;
ind=find(ima1(:)>umbral);

f=2;
for i=1:f:s(1)
for j=1:f:s(2)
for k=1:f:s(3)
  tmp=ima1(i:i+f-1,j:j+f-1,k:k+f-1);  
  nima1((i+f-1)/f,(j+f-1)/f,(k+f-1)/f)=mean(tmp(:));  
end
end
end

% Add noise (2%)
sd=0.02*max(nima1(:));
rima1=nima1+sd*randn(size(nima1));

R=range(rima1(:));

% Denoise ima
tic
nima1=ANLM3D(rima1,sd);
toc

% Reconstruc using Nearest Neighbour
ori=((1+f)/2);
[x,y,z] = meshgrid(ori:f:1-ori+s(2),ori:f:1-ori+s(1),ori:f:1-ori+s(3));
[xi,yi,zi] = meshgrid(1:s(2),1:s(1),1:s(3));
enima1 = interp3(x,y,z,nima1,xi,yi,zi,'nearest'); 

error_nn=sqrt(mean((ima1(ind)-enima1(ind)).^2));
psnr_nn=20*log10(R/error_nn)

% Reconstruc using linear interpolation
bima0 = interp3(x,y,z,nima1,xi,yi,zi,'linear'); 

% do extrapolation (volume borders)
for i=1:floor(f/2)
  bima0(:,:,i) = bima0(:,:,floor(f/2)+1);
  bima0(:,i,:) = bima0(:,floor(f/2)+1,:);
  bima0(i,:,:) = bima0(floor(f/2)+1,:,:);
end
for i=1:floor(f/2)
  bima0(:,:,s(3)-i+1) = bima0(:,:,s(3)-floor(f/2));
  bima0(:,s(2)-i+1,:) = bima0(:,s(2)-floor(f/2),:);
  bima0(s(1)-i+1,:,:) = bima0(s(1)-floor(f/2),:,:);
end

error_lin=sqrt(mean((ima1(ind)-bima0(ind)).^2));
psnr_lin=20*log10(R/error_lin)

% Reconstruc using cubic interpolation
bima1 = interp3(x,y,z,nima1,xi,yi,zi,'cubic'); 

% do extrapolation (volume borders)
for i=1:floor(f/2)
  bima1(:,:,i) = bima1(:,:,floor(f/2)+1);
  bima1(:,i,:) = bima1(:,floor(f/2)+1,:);
  bima1(i,:,:) = bima1(floor(f/2)+1,:,:);
end
for i=1:floor(f/2)
  bima1(:,:,s(3)-i+1) = bima1(:,:,s(3)-floor(f/2));
  bima1(:,s(2)-i+1,:) = bima1(:,s(2)-floor(f/2),:);
  bima1(s(1)-i+1,:,:) = bima1(s(1)-floor(f/2),:,:);
end

error_cubic=sqrt(mean((ima1(ind)-bima1(ind)).^2));
psnr_cubic=20*log10(R/error_cubic)

% Reconstruc using spline interpolation
[x,y,z] = ndgrid(ori:f:1-ori+s(1),ori:f:1-ori+s(2),ori:f:1-ori+s(3));
[xi,yi,zi] = ndgrid(1:s(1),1:s(2),1:s(3));
bima2 = interpn(x,y,z,nima1,xi,yi,zi,'spline'); 

% do extrapolation (volume borders)
for i=1:floor(f/2)
  bima2(:,:,i) = bima2(:,:,floor(f/2)+1);
  bima2(:,i,:) = bima2(:,floor(f/2)+1,:);
  bima2(i,:,:) = bima2(floor(f/2)+1,:,:);
end
for i=1:floor(f/2)
  bima2(:,:,s(3)-i+1) = bima2(:,:,s(3)-floor(f/2));
  bima2(:,s(2)-i+1,:) = bima2(:,s(2)-floor(f/2),:);
  bima2(s(1)-i+1,:,:) = bima2(s(1)-floor(f/2),:,:);
end

error_spline=sqrt(mean((ima1(ind)-bima2(ind)).^2));
psnr_spline=20*log10(R/error_spline)


% Proposed method 
tic
lf=size(ima1)./size(nima1);
[lima]=NLMUpsample2(nima1,lf);
toc

error_proposed=sqrt(mean((ima1(ind)-lima(ind)).^2));
psnr_proposed=20*log10(R/error_proposed)

% draw results

clf
colormap(gray)

ca=s(3)/2;
cs=s(2)/2;
cc=s(1)/2;

subplot(4,4,1)
image(imrotate(rima1(:,:,round(ca/2)),90)/4)
title('LR noisy')
subplot(4,4,2)
image(imrotate(squeeze(rima1(round(cs/2),:,:)),90)/4)
subplot(4,4,5)
image(imrotate(squeeze(rima1(:,round(cc/2),:)),90)/4)

subplot(4,4,3)
image(imrotate(enima1(:,:,ca),90)/4)
title('NN')
subplot(4,4,4)
image(imrotate(squeeze(enima1(cs,:,:)),90)/4)
subplot(4,4,7)
image(imrotate(squeeze(enima1(:,cc,:)),90)/4)

subplot(4,4,9)
image(imrotate(bima2(:,:,ca),90)/4)
title('B-Spline')
subplot(4,4,10)
image(imrotate(squeeze(bima2(cs,:,:)),90)/4)
subplot(4,4,13)
image(imrotate(squeeze(bima2(:,cc,:)),90)/4)

subplot(4,4,11)
image(imrotate(lima(:,:,ca),90)/4)
title('Proposed')
subplot(4,4,12)
image(imrotate(squeeze(lima(cs,:,:)),90)/4)
subplot(4,4,15)
image(imrotate(squeeze(lima(:,cc,:)),90)/4)

save results_noisy;
