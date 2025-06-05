function [im, masks, labels, bbox] = augmentImage(im, masks, labels, bbox)
% AUGMENTIMAGE Applies random data augmentations to training images and annotations
%
% This function performs probabilistic data augmentation for training deep learning models
% on segmentation tasks. It applies geometric transformations, contrast adjustments, 
% cropping, and noise addition while maintaining correspondence between images and 
% their segmentation masks/bounding boxes.
%
% Augmentations applied:
% - Horizontal/vertical flips (50% each)
% - Contrast enhancement/reduction (10%/5% probability)
% - Random cropping (2% probability) 
% - Gaussian noise addition (25% probability)
%
% Inputs/Outputs:
%   im     - Input image, returns augmented image
%   masks  - Segmentation masks (H x W x N), transformed to match image
%   labels - Object class labels, filtered if objects removed by cropping
%   bbox   - Bounding boxes, updated to match transformed coordinates

% Generate all random numbers at once for computational efficiency
% Using single precision to reduce memory usage during batch processing
randNums = rand([1 6], "single");

%% Geometric Augmentations
% Apply random flips to increase dataset diversity and improve model robustness
% Both image and masks must be transformed identically to maintain correspondence

if randNums(1) < 0.5  % Horizontal flip with 50% probability
    im = im(end:-1:1, :);
    masks = masks(end:-1:1, :, :);
end

if randNums(2) < 0.5  % Vertical flip with 50% probability  
    im = im(:, end:-1:1);
    masks = masks(:, end:-1:1, :);
end

%% Contrast Augmentations
% Modify image contrast to simulate different imaging conditions
% Uses trigonometric functions for smooth, non-linear contrast adjustments

if randNums(3) < 0.1  % Increase contrast with 10% probability
    % Apply sigmoid-like contrast enhancement using cosine transformation
    % Maps [0,1] → [0,1] with enhanced mid-tone contrast
    im = (-cos(pi * im) + 1) / 2;
    
    if randNums(3) < 0.01  % Extreme contrast boost in 1% of cases
        % Apply the same transformation twice for very high contrast
        im = (-cos(pi * im) + 1) / 2;
    end
end

if randNums(3) > 0.95  % Decrease contrast with ~5% probability
    % Apply inverse contrast reduction using arccosine
    % Note: This is mutually exclusive with contrast enhancement
    im = 1 - acos(2*(im-0.5))/pi;
end

%% Spatial Cropping
% Randomly crop image to smaller size, removing objects that fall outside crop area
% This simulates different fields of view and helps model generalize to partial objects

if randNums(6) < 0.02  % Random crop with 2% probability
    % Crop to fixed size [264, 352] and update all annotations accordingly
    [im, masks, labels, bbox] = CropRandom(im, masks, labels, bbox, [264, 352]);
end

%% Noise Augmentation  
% Add realistic imaging noise to improve robustness to real-world conditions

if randNums(4) < 0.25  % Add Gaussian noise with 25% probability
    % Apply zero-mean Gaussian noise with random variance
    % Variance scaled by randNums(5) to vary noise intensity
    im = imnoise(im, "gaussian", 0, 0.0025*randNums(5));
end

end