clear
clc
close all

% run image segmentation
load("NewsetNet.mat", 'net')

%net.OverlapThresholdRPN = 0.3;
net.OverlapThresholdPrediction = 0.3;
%net.ScoreThreshold = 0.1;
%%



dsTest= fileDatastore("./MidValDSFs", ReadFcn=@(x)TestIMsMATReader(x));

tic
dsResults= segmentObjects(net, dsTest, "Threshold",0.000001,"MinSize",[2 2],"MaxSize",[80 80],"NumStrongestRegions",1600,"SelectStrongest",true);
toc
%% evaluate



dsResults = fileDatastore("./SegmentObjectResults/", ReadFcn=@(x)SegMATReader(x)); %segmented data
dsTruth  = fileDatastore("./MidValDSFs", ReadFcn=@(x)TestMATReader(x)); %training data
j=1;
i=0.5;
%for i=[0.5:0.05:0.95]
tic
metrics = evaluateInstanceSegmentation(dsResults, dsTruth, i,"Verbose",true);
toc

n="NEWzerothreshmetrics_EFF_"+i+"_.mat";
cellmetrics{j} = metrics;
j=j+1;
save(n)

%end




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
