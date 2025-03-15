clear
clc
close all


addpath('./src');


%trainClassNames = ["CellA"];
imageSize=[260 352 3];

%trainCats = {'CellA'};
trainCats= {'SHSY5Y', 'SKOV3', 'SkBr3', 'Huh7', 'BV2', 'MCF7', 'A172', 'BT474'}




classNames = trainCats;
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];


% datdir="/home/zcemydo/Scratch/TrainV2/";
% ds = fileDatastore([datdir+"/DSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %training datatrainDS = transform(ds, @(x)helper.preprocessData(x, imageSize));


params = createMaskRCNNConfig(imageSize, numClasses, classNames);
params.AnchorBoxes=[ 4 4; 8 8; 16 16; 32 32; 64 64; 128 128; ...
                     4 8; 8 4; 8 16; 16 8; 16 32; 32 16; 64 128; 128 64];

params.NumAnchors = size(params.AnchorBoxes,1);

params.MinSize=[3 3] %stands to reason that any smaller box than 3x3 pixels does not contain enough information to segment, can filter to save on computation
   %  %could probably be increased even further, but computation is not a limiting factor here

   %  params.RPNROIPerImage = params.NumRegionsToSample;


%params.NumStrongestRegionsBeforeProposalNMS;


dlnet = createMaskRCNN(numClasses, params, 'resnet101'); %or resnet 101



imageSize=[260 352 4];
params.ImageSize=imageSize;
disp(params);

%%
save("4ChannelNet.mat", "dlnet", 'params', "imageSize", "trainCats")

