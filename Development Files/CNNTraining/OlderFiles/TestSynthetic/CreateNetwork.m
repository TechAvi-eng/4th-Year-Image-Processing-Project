clear
clc
close all


addpath('./src');


%trainClassNames = ["CellA"];
imageSize=[520 704 3];

trainCats = {'CellA'};
%trainCats= {'CellA', 'SKOV3', 'SkBr3', 'Huh7', 'BV2', 'MCF7', 'A172', 'BT474'}



classNames = trainCats;
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];


% datdir="/home/zcemydo/Scratch/TrainV2/";
% ds = fileDatastore([datdir+"/DSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %training datatrainDS = transform(ds, @(x)helper.preprocessData(x, imageSize));


params = createMaskRCNNConfig(imageSize, numClasses, classNames);




   % %  params.RPNROIPerImage = params.NumRegionsToSample;
   % %  params.MinSize=[3 3] %stands to reason that any smaller box than 3x3 pixels does not contain enough information to segment, can filter to save on computation
   % %  %could probably be increased even further, but computation is not a limiting factor here
   % % 
   %   params.AnchorBoxes=[
   %   32    16
   %  64    32
   % 128    64
   % 256   128
   % 512   256
   %  32    32
   %  64    64
   % 128   128
   % 256   256
   % 512   512
   %  16    32
   %  32    64
   %  64   128
   % 128   256
   % 256   512
   % 8 8
   % 16 8
   % 8 16] %anchor boxes set to reflect cell sizes in cell images, weighted for smaller values
   % 
   % params.NumAnchors=size(params.AnchorBoxes,1) %number of anchor boxes is the number of rows
   % 
   % 
   % params.ROIAlignOutputSize=[20 20];
   % params.MaskOutputSize=[20 20];
   % params.ScaleFactor=[1/20 1/20];
   % params.PositiveOverlapRange=[0.5 1];
   % params.NegativeOverlapRange=[0 0.5];
   % params.NumRegionsToSample=400;
   % params.OverlapThreshold=0.2;
   % params.NumStrongestRegionsBeforeProposalNMS=2000;
   % params.RPNROIPerImage=2000;
   % params.MinSize=[5 5]




dlnet = createMaskRCNN(numClasses, params, 'resnet101'); %or resnet 101

disp(params);

%%

params.ImageSize=[520 704 1]


%%
save("NEWNEWNEW.mat", "dlnet", 'params')

