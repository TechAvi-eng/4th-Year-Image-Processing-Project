% Instance Segmentation Evaluation Script
% This script loads a trained neural network, performs object segmentation 
% on test data, and evaluates the segmentation performance

clear       
clc         
close all   

%% Load Pre-trained Network
% Load the trained neural network from a .mat file
% 'net' contains the segmentation model with pre-configured parameters
load("NetFile.mat", 'net')

%% Network Parameter Configuration
% Configure network thresholds for object detection and segmentation
% net.OverlapThresholdRPN = 0.3;        % RPN overlap threshold (commented out)
net.OverlapThresholdPrediction = 0.3;   % Prediction overlap threshold for NMS
% net.ScoreThreshold = 0.1;             % Minimum confidence score (commented out)

%% Test Data Setup
% Create a file datastore to read test images
% TestIMsMATReader is a custom function to read test image data
dsTest = fileDatastore("./MidValDSFs", ReadFcn=@(x)TestIMsMATReader(x));

%% Object Segmentation
% Perform instance segmentation on the test dataset
tic % Start timing the segmentation process

dsResults = segmentObjects(net, dsTest, ...
    "Threshold", 0.000001, ...          % Very low confidence threshold to capture most objects
    "MinSize", [2 2], ...               % Minimum object size (2x2 pixels)
    "MaxSize", [80 80], ...             % Maximum object size (80x80 pixels)
    "NumStrongestRegions", 1600, ...    % Maximum number of regions to consider
    "SelectStrongest", true);           % Select strongest/most confident regions

toc % End timing and display elapsed time

%% Evaluation Setup
% Set up datastores for evaluation
% dsResults: Contains the segmentation results from the network
dsResults = fileDatastore("./SegmentObjectResults/", ReadFcn=@(x)SegMATReader(x));

% dsTruth: Contains the ground truth annotations for comparison
dsTruth = fileDatastore("./MidValDSFs", ReadFcn=@(x)TestMATReader(x));

%% Evaluation Parameters
% Set IoU (Intersection over Union) threshold for evaluation
% Objects with IoU >= 0.5 with ground truth are considered correct detections
Threshold = 0.5;

%% Performance Evaluation
% Evaluate the instance segmentation performance
tic % Start timing the evaluation process

metrics = evaluateInstanceSegmentation(dsResults, dsTruth, Threshold, ...
    "Verbose", true);  % Enable verbose output to see detailed results

toc % End timing and display elapsed time

% The 'metrics' structure will contain performance measurements such as:
% - Average Precision (AP) at different IoU thresholds
% - Precision and Recall values
% - F1 scores
% - Per-class performance metrics



%%
% function metrics = evaluateInstanceSegmentation(dsResults, dsTruth, threshold)
% 
%     numPred = numel(dsResults);
%     numTruth = numel(dsTruth);
% 
%     % Initialize variables
%     truePositive = 0;
%     iouScores = zeros(numPred, numTruth);
% 
%     % Calculate IoU matrix
%     for i = 1:numPred
%         for j = 1:numTruth
%             iouScores(i, j) = calculateIoU(dsResults{i}, dsTruth{j});
%         end
%     end
% 
%     % Match predictions to ground truth based on threshold
%     matched = iouScores > threshold;
% 
%     % Determine true positives
%     for i = 1:numTruth
%         if any(matched(:, i))
%             truePositive = truePositive + 1;
%         end
%     end
% 
%     % Calculate false positives and false negatives
%     falsePositive = numPred - truePositive;
%     falseNegative = numTruth - truePositive;
% 
%     % Metrics calculations
%     precision = truePositive / (truePositive + falsePositive);
%     recall = truePositive / (truePositive + falseNegative);
%     f1Score = 2 * (precision * recall) / (precision + recall);
%     meanIoU = mean(iouScores(matched));
% 
%     % Output metrics
%     metrics = struct();
%     metrics.Precision = precision;
%     metrics.Recall = recall;
%     metrics.F1Score = f1Score;
%     metrics.MeanIoU = meanIoU;
% end
% 
% function iou = calculateIoU(mask1, mask2)
%     % Calculate Intersection over Union (IoU) for two binary masks.
%     % Inputs:
%     %   mask1 - Binary mask for prediction
%     %   mask2 - Binary mask for ground truth
%     % Output:
%     %   iou - Intersection over Union score
% 
%     intersection = sum(mask1(:) & mask2(:));
%     union = sum(mask1(:) | mask2(:));
%     iou = intersection / union;
% end
% 
