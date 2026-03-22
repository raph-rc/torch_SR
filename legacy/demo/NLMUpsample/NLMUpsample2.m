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
%    Usage of NLMUpsample:
%  
%    fima=NLMUpsample(ima,f)
%
%    ima: LR volume
%    f: Magnification factor in each dimension (for example [2,2,2])
%    fima: HR upsampled volume
%
%**************************************************************************
       
function [lima]=NLMUpsample2(ima,lf)

disp('Processing...');

% fixed range
m=max(ima(:));
ima=ima*256/m;

% Initial interpolation
bima=InitialInterpolation(ima,lf);

% Parameters 
sigma=stdfilt(bima,ones(3,3,3));
pad=padarray(sigma,[1,1,1],'symmetric');
sigma=convn(pad,ones(3,3,3),'valid')/(3*3*3);
level=sigma/2;              
tol=1.2;%0.01*mean(sigma(:));             
v=3;                        
F=bima;                     
last=F;
ii=1;
iii=1;

% Iterative reconstruction
down=0;
while(1)

%tic
lima=cMRegularizarNLM3D_V2(F,v,1,level,lf);
%t1=toc

d(ii)=mean(abs(F(:)-lima(:)))

if(ii>1)
  if((d(ii-1)/d(ii))<tol && down==0) 
    down=1;  
    level=level/2;
    ds(iii)=mean(abs(last(:)-lima(:)));
    if(iii>1)
      if(ds(iii-1)/ds(iii)<tol) break; end; 
    end
    last=lima;
    iii=iii+1;  
  else
    down=0;
  end
  if(d(ii)<=0.001) break; end;
end

F=lima;
ii=ii+1;

end

lima=lima*m/256;

disp('Done!');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [bima]=InitialInterpolation(nima1,lf)

s=size(nima1).*lf;
ori=((1+lf)/2);

% reconstruc using spline interpolation
[x,y,z] = ndgrid(ori(1):lf(1):1-ori(1)+s(1),ori(2):lf(2):1-ori(2)+s(2),ori(3):lf(3):1-ori(3)+s(3));
[xi,yi,zi] = ndgrid(1:s(1),1:s(2),1:s(3));
bima2 = interpn(x,y,z,nima1,xi,yi,zi,'spline'); 

% deal with extreme slices
for i=1:floor(lf(1)/2)
  bima2(i,:,:) = bima2(floor(lf(1)/2)+1,:,:);
end
for i=1:floor(lf(2)/2)
  bima2(:,i,:) = bima2(:,floor(lf(2)/2)+1,:);
end
for i=1:floor(lf(3)/2)
  bima2(:,:,i) = bima2(:,:,floor(lf(3)/2)+1);
end

for i=1:floor(lf(1)/2)
  bima2(s(1)-i+1,:,:) = bima2(s(1)-floor(lf(1)/2),:,:);
end
for i=1:floor(lf(2)/2)
  bima2(:,s(2)-i+1,:) = bima2(:,s(2)-floor(lf(2)/2),:);  
end
for i=1:floor(lf(3)/2)
  bima2(:,:,s(3)-i+1) = bima2(:,:,s(3)-floor(lf(3)/2));
end

% mean correction
for i=1:lf(1):s(1)
for j=1:lf(2):s(2)
for k=1:lf(3):s(3)  
    tmp=bima2(i:i+lf(1)-1,j:j+lf(2)-1,k:k+lf(3)-1);  
    off=nima1((i+lf(1)-1)/lf(1),(j+lf(2)-1)/lf(2),(k+lf(3)-1)/lf(3))-mean(tmp(:));
    bima(i:i+lf(1)-1,j:j+lf(2)-1,k:k+lf(3)-1)=bima2(i:i+lf(1)-1,j:j+lf(2)-1,k:k+lf(3)-1)+off;
end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
