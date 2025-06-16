function [imageOut maskOut BBoxOut] = resizeImageandMask(imageIn, maskIn, BBoxIn, NewSize)
% RESIZEIMAGEMASK Resizes images and masks to target dimensions using center padding
%
% This function standardizes image dimensions for neural network training by padding
% images to the target size while maintaining aspect ratio and centering content.
% It simultaneously updates segmentation masks and bounding box coordinates to
% maintain annotation consistency after spatial transformation.
%
% The function uses zero-padding rather than stretching to avoid distortion,
% which is crucial for maintaining accurate spatial relationships in segmentation tasks.
%
% Inputs:
%   imageIn  - Input image of any size
%   maskIn   - Segmentation masks (H x W x N), can be empty []
%   BBoxIn   - Bounding boxes in [x y width height] format  
%   NewSize  - Target dimensions [height, width]
%
% Outputs:
%   imageOut - Padded image with dimensions NewSize
%   maskOut  - Padded masks with same spatial transformation
%   BBoxOut  - Bounding boxes with coordinates adjusted for padding offset

% Get original image dimensions for padding calculations
sz = size(imageIn);
paddedH = NewSize(1);
paddedW = NewSize(2);

% Alternative sizing approach for networks requiring specific divisibility
% Commented out: ensures dimensions are divisible by 16 (common CNN requirement)
% paddedH = int32(ceil(sz(1)/16)*16);
% paddedW = int32(ceil(sz(2)/16)*16);

%% Calculate Center Padding Coordinates
% Determine where to place original image within padded canvas for centering
% c1: row indices for vertical centering
% c2: column indices for horizontal centering
c1 = ceil((paddedH - sz(1))/2)+1:ceil((paddedH + sz(1))/2);
c2 = ceil((paddedW - sz(2))/2)+1:ceil((paddedW + sz(2))/2);

%% Update Bounding Box Coordinates
% Adjust bounding box positions to account for padding offset
if ~isempty(BBoxIn)
    BBoxOut = BBoxIn;
    BBoxOut(:,2) = min(BBoxIn(:,2)+(NewSize(1)-sz(1))/2, NewSize(1));
else
    BBoxOut = [];
end


%% Pad Image with Center Alignment
% Create padded canvas and place original image at center
% Preserves data type and handles multi-channel images
imageOut = zeros(paddedH, paddedW, size(imageIn, 3), 'like', imageIn);
imageOut(c1, c2, :) = imageIn;

%% Handle Segmentation Masks
% Apply identical spatial transformation to masks to maintain correspondence
if ~isempty(maskIn)
    % Pad masks using same center alignment as image
    maskOut = zeros(paddedH, paddedW, size(maskIn, 3), 'like', maskIn);
    maskOut(c1, c2, :) = maskIn;
else
    % Handle case where no masks are provided (e.g., inference mode)
    maskOut = [];
end

end