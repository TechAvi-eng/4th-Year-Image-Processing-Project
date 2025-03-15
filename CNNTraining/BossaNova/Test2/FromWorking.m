clear
clc
close all

load("Res101-new.mat", 'net');

%%

ds = fileDatastore("../SmallDSFs/", ReadFcn=@(x)MATReader1C(x, 0)); %training data

%%
options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=100, ...
    LearnRateDropFactor=0.5, ...
    Plot="none", ...  
    MaxEpochs=20, ...
    MiniBatchSize=1, ...
    ResetInputNormalization=false, ...
    ExecutionEnvironment="cpu", ...
    VerboseFrequency=1, ...
    L2Regularization=1e-5) 

%%
[net,info] = trainMRCNN(ds,net,options, NumStrongestRegions=2000, NumRegionsToSample=100, PositiveOverlapRange=[0.5 1], NegativeOverlapRange=[0 0.5], ForcedPositiveProposals=false, FreezeSubNetwork="backbone")


%%
max(net.RegionProposalNet.Layers(3,1).Weights, [], 'all')


%%
if ~exist("im")
    load("../SingleDS/label_A172_Phase_A7_1_00d00h00m_2.tif.mat", "im")
    im=rescale(im);
    im=repmat(im ,[1 1 1]); 
end


tic
for i = 1
[masks,labels,scores,boxes] = segmentObjects(net,im,Threshold=0.5,NumStrongestRegions=5000, SelectStrongest=true, MinSize=[4 4],MaxSize=[80 80] );

if(isempty(masks))
    overlayedImage = im(:,:,1);
else
    overlayedImage = insertObjectMask(im(:,:,1), masks,Color=lines(size(masks, 3)) );
end
end

figure, imshow(overlayedImage)

%showShape("rectangle", gather(boxes), "Label", scores, "LineColor",'r')