clear
clc 
close all

addpath("src/")

imageSize = [520/2 704/2 3];
% ds = fileDatastore(unpackAnnotationFolder, 'ReadFcn',@(x)cocoAnnotationMATReader(x, trainImgFolder));
% trainDS = transform(ds, @(x)helper.preprocessData(x, imageSize));
% trainDS.shuffle();
% data = preview(trainDS)


%classNames = {'CellA'};
classNames= {'SHSY5Y', 'SKOV3', 'SkBr3', 'Huh7', 'BV2', 'MCF7', 'A172', 'BT474'}
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];

params = createMaskRCNNConfig(imageSize, numClasses, classNames);

params.PositiveOverlapRange=[0.5 1];
params.NegativeOverlapRange=[0 0.5];
params.OverlapThreshold=0.6;


params.AnchorBoxes=params.AnchorBoxes/2;

params.NumAnchors = size(params.AnchorBoxes,1);
params.StandardizeRegressionTargets = false;


disp(params);

dlnet = createMaskRCNN(numClasses, params, 'resnet101');

params.ImageSize=[520/2 704/2 1];