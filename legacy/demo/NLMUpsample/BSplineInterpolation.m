function [bima2]=BSplineInterpolation(nima1,lf)

s=size(nima1).*lf;
ori=((1+lf)/2);

% reconstruc using spline interpolation
[x,y,z] = ndgrid(ori(1):lf(1):1-ori(1)+s(1),ori(2):lf(2):1-ori(2)+s(2),ori(3):lf(3):1-ori(3)+s(3));
[xi,yi,zi] = ndgrid(1:s(1),1:s(2),1:s(3));
bima2 = interpn(x,y,z,nima1,xi,yi,zi,'spline'); 

% do extrapolation (volume borders)
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

