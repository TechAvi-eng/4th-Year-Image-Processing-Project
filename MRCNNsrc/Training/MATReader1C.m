function out = MATReader1C(filename, augmentOnOff)
% MATREADER1C Loads and preprocesses training data from MAT files for deep learning
%
% This function serves as a data loader for training segmentation models. It loads
% annotated image data from MAT files, applies consistent preprocessing, and 
% optionally performs data augmentation. The function standardizes all inputs
% to the required dimensions and format for network training.
%
% Processing pipeline:
% 1. Load image and annotations from MAT file
% 2. Normalize image intensities to [0,1] range
% 3. Resize image and annotations to standard dimensions [528, 704]
% 4. Apply random augmentations if enabled
% 5. Package outputs in cell array format
%
% Inputs:
%   filename     - Path to MAT file containing 'im', 'masks', 'bbox', 'label' variables
%   augmentOnOff - Boolean flag (1=apply augmentation, 0=preprocessing only)
%
% Output:
%   out - Cell array containing: {image, bounding_boxes, labels, masks}

%% Load Training Data
% Load MAT file containing pre-annotated training sample
% Expected variables: im (image), masks (segmentation), bbox (bounding boxes), label (classes)
load(filename);

%% Preprocessing Pipeline
% Normalize image intensities to [0,1] range for consistent network input
im = rescale(im);

% Standardize dimensions to required network input size [528, 704]
% This ensures all training samples have consistent dimensions
[im, masks, bbox] = resizeImageandMask(im, masks, bbox, [528, 704]);

%% Optional Data Augmentation
% Apply random augmentations during training to improve model generalization
% Augmentations include flips, contrast changes, cropping, and noise addition
if augmentOnOff == 1
    [im, masks, label, bbox] = augmentImage(im, masks, label, bbox);
end

%% Package Output for Training Pipeline
% Return data in cell array format expected by training framework
out{1} = im;      % Preprocessed image
out{2} = bbox;    % Bounding boxes (updated if augmented)
out{3} = label;   % Class labels (filtered if objects removed)
out{4} = masks;   % Segmentation masks (transformed to match image)

end