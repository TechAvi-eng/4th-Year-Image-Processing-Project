clear
clc
close all

load("File2.mat", 'net')

net.OverlapThresholdRPN = 0.3;
net.OverlapThresholdPrediction = 0.3;
%net.ScoreThreshold = 0.1;
%%



dsTest= fileDatastore("./ExampleDS/", ReadFcn=@(x)TestIMsMATReader(x));

tic
dsResults= segmentObjects(net, dsTest, "Threshold",0.001,"MinSize",[2 2],"NumStrongestRegions",Inf,"SelectStrongest",true);
toc
%% evaluate



dsResults = fileDatastore("./SegmentObjectResults1/", ReadFcn=@(x)SegMATReader(x)); %segmented data
dsTruth  = fileDatastore("./ExampleDS", ReadFcn=@(x)TestMATReader(x)); %training data
j=1;
 %for i=[0.5:0.05:0.95]
 i=0.5;
tic
metrics = evaluateInstanceSegmentation(dsResults, dsTruth, i,"Verbose",true);
toc


%save(n)
%end





