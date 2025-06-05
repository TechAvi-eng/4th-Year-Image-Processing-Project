% Cell Segmentation Analysis and Evaluation Script
% This script loads a trained Mask R-CNN model, performs cell segmentation 
% on validation data, and generates performance analysis plots

clear       
clc         
close all   

% Add custom Mask R-CNN source code to MATLAB path
addpath(genpath("~/Scratch/MRCNNsrc"));

%% Load Pre-trained Network
% Load the trained Mask R-CNN model for cell segmentation
load('NewestNet.mat', 'net');

%% Network Parameter Configuration
% Set low overlap thresholds to be more sensitive to overlapping objects
net.OverlapThresholdRPN = 0.1;        % Region Proposal Network overlap threshold
net.OverlapThresholdPrediction = 0.1; % Final prediction overlap threshold

%% Data File Discovery
% Get list of all files in the validation dataset directory
di = dir('./MidValDSFs');

% Remove empty files (directories and zero-byte files)
dirb = vertcat(di.bytes);    % Get file sizes
di(dirb==0, :) = [];         % Remove entries with zero bytes
diname = {di.name};          % Extract filenames

%% Segmentation Parameters
Thr = 1e-5;  % Very low segmentation threshold to capture weak detections

%% Batch Processing Loop
% Process each validation image through the segmentation network
for i = [1:length(diname)]
    % Load validation data (contains 'im' image and 'bbox' ground truth)
    load(strcat('./MidValDSFs/', diname{i}));
    
    % Preprocessing: normalize image intensities to [0,1] range
    im = rescale(im);
    
    % Resize image to standard network input size (528x704 pixels)
    % Masks are not needed for inference, hence empty []
    [im, ~] = resizeImageandMask(im, [], [528, 704]);
    
    % Perform cell segmentation using the trained network
    [pmasks, plabels, pscores, pboxes] = segmentCells(net, im, ...
        "SegmentThreshold", Thr, ...        % Detection confidence threshold
        "NumstrongestRegions", 1000);       % Maximum regions to consider
    
    % Store results for each image
    allpboxes{i} = pboxes;      % Predicted bounding boxes
    allpsocres{i} = pscores;    % Prediction confidence scores  
    allabbox{i} = bbox;         % Ground truth bounding boxes
    
    % Display progress
    i
end

%% Performance Evaluation
% Calculate precision-recall curve and average precision
% Uses IoU threshold of 0.075 for matching predictions to ground truth
[precision, recall, ap] = calculate_combined_pr_curve(allpboxes, allpsocres, allabbox, 0.075);

%% Cell Count Analysis
% Compare predicted vs actual cell counts for each image
for i = 1:569
    % Count predictions above confidence threshold of 0.2
    numpred(i) = nnz(allpsocres{i} > 0.2);
    % Alternative: count all predictions regardless of score
    % numpred(i) = size(allpboxes{i}, 1);
    
    % Count ground truth cells
    numac(i) = size(allabbox{i}, 1);
end

%% Generate Cell Count Correlation Plot
close all
% Create scatter plot comparing true vs predicted cell counts
scatter(numac, numpred, 32, 'filled', 'ko')
xlabel('True Number of Cells', 'interpreter', 'latex')
ylabel('Predicted Number of Cells', 'interpreter', 'latex')

hold on
% Add perfect correlation line (y = x)
plot([0 1500], [0, 1500], 'linestyle', '--', 'linewidth', 2, 'color', [1 1 1]*0.4)

% Configure plot appearance
axis equal   
xlim([0 1200])   
ylim([0 1200])   
box on   
grid on        
%fontname('CMU Serif') 
fontsize(16, 'points')    

%% Generate Precision-Recall Curve Plot
close all
% Plot precision-recall curve with endpoints for complete curve
plot([recall; 1], [precision; 0], 'LineWidth', 3, 'Color', 'k')
xlabel('Recall', 'interpreter', 'latex')
ylabel('Precision', 'interpreter', 'latex')
title('Precision-Recall Curve at IoU=0.5')

hold on
% Configure plot appearance
axis equal 
xlim([0 1]) 
ylim([0 1]) 
box on  
grid on  
%fontname('CMU Serif')       