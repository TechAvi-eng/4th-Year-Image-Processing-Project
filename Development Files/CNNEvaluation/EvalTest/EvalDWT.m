clear
clc
close all

% run image segmentation
load("TestDWTNet.mat")


%%
tic
dsTest= fileDatastore("SmallValDSFs/val1.tif.mat", ReadFcn=@(x)DWTTestIMsMATReader(x));
dsResults= segmentObjects(net, dsTest, "Threshold",0.25,"MinSize",[4 4],"MaxSize",[60 60],"NumStrongestRegions",inf,"SelectStrongest",true);
toc
% evaluate

%%
clc

dsResults = fileDatastore("./SegmentObjectResults/", ReadFcn=@(x)DWTSegMATReader(x)); %segmented data
dsTruth  = fileDatastore("./SmallValDSFs/val1.tif.mat", ReadFcn=@(x)DWTTestMATReader(x)); %training data

metrics = evaluateInstanceSegmentation(dsResults, dsTruth, 0.5);
%save("metrics101-0.75.mat")


