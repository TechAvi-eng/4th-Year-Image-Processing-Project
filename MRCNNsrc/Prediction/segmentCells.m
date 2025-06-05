function [masks labels scores boxes] = segmentCells(net, Image, Options)
% SEGMENTCELLS Performs cell segmentation using optional DWT denoising and Mask R-CNN
%
% This function provides a complete pipeline for cell segmentation in microscopy images:
% 1. Optional DWT-based denoising to reduce noise while preserving cell boundaries
% 2. Image preprocessing (rescaling to [0,1] range)
% 3. Mask R-CNN inference for instance segmentation
% 4. Optional visualization of results with masks and confidence scores
%
% Outputs:
%   masks  - Binary masks for each detected cell (H x W x N)
%   labels - Class labels for each detection
%   scores - Confidence scores for each detection [0,1]
%   boxes  - Bounding boxes in [x y width height] format

arguments % Input validation and default parameter values
    net MRCNN                                                              % Pre-trained Mask R-CNN network
    Image = [];                                                            % Input image to segment
    Options.Denoise (1,1) logical = 1;                                    % Enable/disable DWT denoising preprocessing
    Options.Wavelet char = 'db5'                                          % Wavelet type for denoising (Daubechies 5)
    Options.Level (1,1) {mustBeInteger, mustBeReal} = 4                   % Decomposition levels for DWT denoising
    Options.DWTThreshold {mustBeGreaterThanOrEqual(Options.DWTThreshold, 0), ...
                         mustBeLessThan(Options.DWTThreshold, 1), ...
                         mustBeReal(Options.DWTThreshold)} = 0.02;         % Threshold factor for DWT denoising
    Options.SegmentThreshold (1,1) {mustBeGreaterThanOrEqual(Options.SegmentThreshold, 0), ...
                                   mustBeLessThan(Options.SegmentThreshold, 1), ...
                                   mustBeReal(Options.SegmentThreshold)} = 0.5;  % Confidence threshold for accepting detections
    Options.NumstrongestRegions (1,1) = 5000;                            % Max number of region proposals from RPN
    Options.SelectStrongest logical = 1;                                   % Whether to select only strongest detections
    Options.MinSize (1,2) = [2 2];                                       % Minimum object size [height width] in pixels
    Options.MaxSize (1,2) = [200 200];                                   % Maximum object size [height width] in pixels
    Options.ShowMasks (1,1) logical = 0;                                 % Display segmentation masks overlay
    Options.ShowScores (1,1) logical = 0;                                % Display bounding boxes with confidence scores
    Options.ExecutionEnvironment char = 'cpu';                           % Computation device: 'cpu' or 'gpu'
end

%% Preprocessing: Optional Denoising
% Apply DWT-based denoising if enabled to reduce noise while preserving cell edges
if Options.Denoise == 1
    Image = DWT_Denoise(Image, "Level", Options.Level, "Threshold", Options.DWTThreshold, "Wavelet", Options.Wavelet);
end

%% Image Normalization for Neural Network Input
% Rescale image intensities to [0,1] range as required by Mask R-CNN
Image = rescale(Image);

%% Mask R-CNN Inference
% Perform instance segmentation using the pre-trained network
% Returns masks, class labels, confidence scores, and bounding boxes
[masks, labels, scores, boxes] = segmentObjects(net, Image, ...
    Threshold=Options.SegmentThreshold, ...
    NumStrongestRegions=Options.NumstrongestRegions, ...
    SelectStrongest=Options.SelectStrongest, ...
    MinSize=Options.MinSize, ...
    MaxSize=Options.MaxSize, ...
    ExecutionEnvironment=Options.ExecutionEnvironment);

%% Visualization (Optional)
% Display segmentation results if requested
if Options.ShowMasks == 1
    % Create overlay image with colored masks
    if(isempty(masks))
        % No detections found - show original image
        overlayedImage = Image(:,:,1);
    else
        % Overlay masks with different colors for each detected cell
        overlayedImage = insertObjectMask(Image(:,:,1), masks, Color=lines(size(masks, 3)));
    end
    
    % Display the overlayed image
    figure, imshow(overlayedImage)
    
    % Optionally add bounding boxes and confidence scores
    if Options.ShowScores == 1
        % Draw bounding boxes with confidence scores as labels
        showShape("rectangle", gather(boxes), "Label", scores, "LineColor", 'r');
    end
end

end