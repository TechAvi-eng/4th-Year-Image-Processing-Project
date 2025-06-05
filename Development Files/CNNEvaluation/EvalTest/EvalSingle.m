clear
clc
close all

% run image segmentation
load("File2.mat", 'net')

net.OverlapThresholdRPN = 0.1;
net.OverlapThresholdPrediction = 0.1;
%net.ScoreThreshold = 0.1;
%%



dsTest= fileDatastore("./SingleDS/", ReadFcn=@(x)TestIMsMATReader(x));

tic
dsResults= segmentObjects(net, dsTest, "Threshold",0.001,"MinSize",[2 2],"NumStrongestRegions",Inf,"SelectStrongest",true);
toc
%% evaluate



dsResults = fileDatastore("./SegmentObjectResults5/", ReadFcn=@(x)SegMATReader(x)); %segmented data
dsTruth  = fileDatastore("./SingleDS", ReadFcn=@(x)TestMATReader(x)); %training data
j=1;
i=0.5;
for i=[0.5:0.05:0.95]
tic
metrics = evaluateInstanceSegmentation(dsResults, dsTruth, i,"Verbose",true);
toc
cellmetrics{j} = metrics;
Ap(j)=metrics.ImageMetrics.AP;
j=j+1;
%save(n)
end





