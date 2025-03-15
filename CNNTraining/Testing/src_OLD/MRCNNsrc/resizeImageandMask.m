function [imageOut maskOut] = resizeImageandMask(imageIn, maskIn, InputSize)



sz = size(imageIn);
paddedH = InputSize(1);
paddedW = InputSize(2);

% alternative: make image divisible by 16
% paddedH = int32(ceil(sz(1)/16)*16);
% paddedW = int32(ceil(sz(2)/16)*16);

imageOut = zeros(paddedH,paddedW,size(imageIn, 3), 'like', imageIn);
imageOut( ceil((paddedH - sz(1))/2)+1:ceil((paddedH + sz(1))/2), ceil((paddedW - sz(2))/2 )+1:ceil((paddedW + sz(2))/2 ), :) = imageIn; %pad equally from all sides


%%
maskOut = zeros(paddedH,paddedW,size(maskIn, 3), 'like', maskIn);
maskOut( ceil((paddedH - sz(1))/2)+1:ceil((paddedH + sz(1))/2), ceil((paddedW - sz(2))/2 )+1:ceil((paddedW + sz(2))/2 ), :) = maskIn; %pad equally from all sides
