function [im, masks, labels, bbox] = CropRandom(im, masks, labels, bbox, cropSize)
% CROPRANDOM Randomly crops image and annotations, then resizes back to standard dimensions
%
% This function performs random spatial cropping for data augmentation while maintaining
% annotation consistency. It crops to a smaller size at a random location, removes
% objects that are completely cropped out, then resizes back to the standard size.
% This simulates different fields of view and zoom levels during training.
%
% Process:
% 1. Generate random crop position within valid bounds
% 2. Extract cropped region from image and masks  
% 3. Remove objects that fall completely outside crop area
% 4. Resize cropped content back to standard dimensions [528, 704]
% 5. Update corresponding labels and bounding boxes
%
% Inputs:
%   im       - Input image (assumed to be 528x704 based on offset calculations)
%   masks    - Segmentation masks (H x W x N)
%   labels   - Object class labels (N x 1)
%   bbox     - Bounding boxes (N x 4)
%   cropSize - Desired crop dimensions [height, width]

% Generate random crop position
randNums = rand([1 2], 'single');

% Calculate random offset ensuring crop stays within image bounds
% Assumes input image is 528x704 with 4-pixel padding consideration
% offset(1): random row position from 4 to (520-cropSize(1)+4)  
% offset(2): random column position from 0 to (704-cropSize(2))
offset = ceil([randNums(1)*(520-cropSize(1))+4, randNums(2)*(704-cropSize(2))]);

%% Extract Cropped Region and Filter Empty Objects
% Crop masks to specified region
masks2 = masks(offset(1):offset(1)+cropSize(1)-1, offset(2):offset(2)+cropSize(2)-1, :);

% Identify masks that are completely cropped out (all zeros)
% This prevents invalid annotations for objects no longer visible
ind = squeeze(all(masks2==0, [1,2]));

% Remove empty masks
masks2 = masks2(:,:,~ind);

% Safety check: avoid cropping if no objects remain in cropped region
if isempty(masks2)
    return % Return original inputs unchanged
end

%% Update All Annotations Consistently
% Replace masks with cropped and filtered version
masks = masks2;

% Remove corresponding bounding boxes for cropped-out objects
bbox = bbox(~ind,:);

% Resize masks back to standard dimensions for network compatibility
masks = imresize(masks, [528, 704], 'bilinear');

% Remove labels for objects that were cropped out
labels = labels(~ind);

%% Process Image with Same Transformations
% Apply identical cropping to image
im = im(offset(1):offset(1)+cropSize(1)-1, offset(2):offset(2)+cropSize(2)-1);

% Resize image back to standard dimensions to maintain consistent input size
im = imresize(im, [528, 704], 'bilinear');

end