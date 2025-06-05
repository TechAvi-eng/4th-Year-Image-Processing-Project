function [imageOut maskOut] = resizeImageandMask(imageIn, maskIn, InputSize)



sz = size(imageIn);
paddedH = InputSize(1);
paddedW = InputSize(2);

% alternative: make image divisible by 16
% paddedH = int32(ceil(sz(1)/16)*16);
% paddedW = int32(ceil(sz(2)/16)*16);

c1 = ceil((paddedH - sz(1))/2)+1:ceil((paddedH + sz(1))/2 );
c2 = ceil((paddedW - sz(2))/2 )+1:ceil((paddedW + sz(2))/2 );

imageOut = zeros(paddedH,paddedW,size(imageIn, 3), 'like', imageIn);
imageOut(c1, c2, :) = imageIn; %pad equally from all sides


%%
if ~isempty(maskIn)
maskOut = zeros(paddedH,paddedW,size(maskIn, 3), 'like', maskIn);
maskOut(c1, c2, :) = maskIn; %pad equally from all sides
else
    maskOut = [];
end
