clear
clc
close all

%%
addpath("./MRCNNsrc") %when running on Myriad, needed to load the vision
%toolbox

%%
ds = fileDatastore("./SingleDS/", ReadFcn=@(x)MATReaderDWT(x, 1)); %training data
data = preview(ds)


%%
trainClassNames = ["CellA"];
imageSizeTrain = [528/2 704/2 4];

ABs = [20 20; 20 40; 40 20]*2;
ABs = [ABs; ABs*2; ABs*4; ABs*8; ABs*16]/2;
NetDataDir = "./NetDataRes50";
 
net = MRCNN(trainClassNames,ABs, NetDataDir,InputSize=imageSizeTrain, ScaleFactor=[1 1]/16)


%%
options = trainingOptions("adam", ...
    InitialLearnRate=0.00025, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=100, ...
    LearnRateDropFactor=0.1, ...
    Plot="none", ...  
    MaxEpochs=15, ...
    MiniBatchSize=1, ...
    BatchNormalizationStatistics="moving", ...
    ResetInputNormalization=false, ...
    ExecutionEnvironment="auto", ...
    CheckpointPath="./", ...
    CheckpointFrequency=1000, ...
    VerboseFrequency=1, ...
    GradientThreshold=2)


%%
 

[net,info] = trainMRCNN(ds,net,options, NumStrongestRegions=inf, NumRegionsToSample=256, PositiveOverlapRange=[0.7 1], NegativeOverlapRange=[0 0.3], ForcedPositiveProposals=true) 


%%
load("SingleDS/label_A172_Phase_A7_1_00d00h00m_2.tif.mat", "im")
im=rescale(im);
im=repmat(im ,[1 1 1]);

%%
im=rand([520 704]);
%%
tic
% %% utility to use model to te
% st certain images 
% im1=imread("../JSON_FORMATTING/LiveCellsIms1/livecell_test_images/A172_Phase_C7_1_00d00h00m_3.tif");
% 
%net.ProposalsOutsideImage='clip';
     [masks,labels,scores,boxes] = segmentObjects(net,im,Threshold=0.1,NumStrongestRegions=Inf, SelectStrongest=true, MinSize=[3 3],MaxSize=[80 80]);
% 
% %%
% imshow(insertObjectMask(im1,masks, Color=lines(size(masks, 3))))
toc

if(isempty(masks))
    overlayedImage = im(:,:,1);
else
    overlayedImage = insertObjectMask(im(:,:,1), masks,Color=lines(size(masks, 3)) );
end

figure, imshow(overlayedImage)

% Show the bounding boxes and labels on the objects
showShape("rectangle", gather(boxes), "Label", scores, "LineColor",'r')


%%
X=dlarray(single(im), 'SSCB');
out=predict(dlnetFeature, X);
