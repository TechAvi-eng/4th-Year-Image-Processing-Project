clear
clc
close all


addpath('./src');

%%


%trainClassNames = ["CellA"];
imageSize=[520 704 3];

trainCats = {'CellA'};


classNames = trainCats;
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];


% datdir="/home/zcemydo/Scratch/TrainV2/";
% ds = fileDatastore([datdir+"/DSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %training datatrainDS = transform(ds, @(x)helper.preprocessData(x, imageSize));

datdir="..";
ds = fileDatastore([datdir+"/SmallDSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %training datatrainDS = transform(ds, @(x)helper.preprocessData(x, imageSize));

trainDS = transform(ds, @(x)helper.preprocessData(x, imageSize));


data = preview(trainDS) 

params = createMaskRCNNConfig(imageSize, numClasses, classNames);


    params.NumRegionsToSample = 1000;
    params.NumStrongestRegions = 2000;
    params.RPNROIPerImage=1000;

    params.AnchorBoxes=[
    32    16
    64    32
   8    4
   20   40
   30   15
    32    32
    64    64
   20   20
   40   40
   60   60
    16    32
    32    64
    64   128
   15   30
   20   40
   10 30
   20 60
   30 90
   30 10
   60 20
   90 30
   128 128
   20 20
   30 30
   15 15
   10 10
   5 15
   15 5
   5 5]

        params.NumAnchors=size(params.AnchorBoxes,1)


disp(params);


dlnet = createMaskRCNN(numClasses, params, 'resnet101'); %or resnet 50/101