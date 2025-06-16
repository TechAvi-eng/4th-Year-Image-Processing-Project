clear
clc
close all
%%
clc

trainClassNames = ["CellA"];
imageSizeTrain = [528 704 1];

ABs = [21 21; 21 32; 32 21;...
    32 32; 47 32; 32 47;...
    47 47; 71 47; 47 71;...
    71 71; 107 71; 71 107;...
    107 107; 160 107; 107 160; ...
    160 160];

net = CascadeRCNN(trainClassNames,ABs,InputSize=imageSizeTrain, ScaleFactor=[1 1]/16,ModelName='EfficientNet')


%%
ds = fileDatastore("./SingleDS/", ReadFcn=@(x)MATReader1C(x, 0)); %training data
data = preview(ds)

options = trainingOptions("adam", ... 
    InitialLearnRate=0.00001, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=300, ...
    LearnRateDropFactor=0.1, ...
    Plot="none", ...  
    MaxEpochs=320, ...
    MiniBatchSize=2, ...
    ResetInputNormalization=false, ...
    ExecutionEnvironment="cpu", ...
    VerboseFrequency=1, ...
    L2Regularization=1e-4, ...
    GradientThreshold=1, ...
    BatchNormalizationStatistics="moving")

%%

[net,info] = trainCascadeRCNN(ds,net,options, NumStrongestRegions=150, NumRegionsToSample=150, PositiveOverlapRange=[0.5 1], NegativeOverlapRange=[0 0.5], ForcedPositiveProposals=false)


%%
load("SingleDS/label_A172_Phase_A7_1_00d00h00m_2.tif.mat", "im")
im=rescale(im);
im=repmat(im ,[1 1 1]); 


im = smartResize(im, [528, 704]);


%%
clc
tic
% %% utility to use model to test certain images 
%net.ProposalsOutsideImage='clip';
%net.MinScore = 0.001;
     [masks,labels,scores,boxes] = segmentObjects(net,im,Threshold=0.005,NumStrongestRegions=1000, SelectStrongest=true, MinSize=[1 1],MaxSize=[80 80] );
toc

%scores = 1./(1+exp(-scores));
%  
% %%
% imshow(insertObjectMask(im1,masks, Color=lines(size(masks, 3))))

if(isempty(masks))
    overlayedImage = im(:,:,1);
else
    overlayedImage = insertObjectMask(im(:,:,1), masks,Color=lines(size(masks, 3)) );
end

figure, imshow(overlayedImage)

% Show the bounding boxes and labels on the objects
%showShape("rectangle", gather(boxes), "Label", scores, "LineColor",'r')


%%
dlX = dlarray(im, 'SSCB');
 dlFeatures = predict(net.FeatureExtractionNet, dlX, 'Acceleration','auto');
    
            [dlRPNScores, dlRPNReg] = predict(net.RegionProposalNet, dlFeatures, 'Outputs',{'RPNClassOut', 'RPNRegOut'});


%%
tic
% %% utility to use model to test certain images 
% im1=imread("../JSON_FORMATTING/LiveCellsIms1/livecell_test_images/A172_Phase_C7_1_00d00h00m_3.tif");
% 

%%
% imshow(insertObjectMask(im1,masks, Color=lines(size(masks, 3))))

if(isempty(masks))
    overlayedImage = im(:,:,1);
else
    overlayedImage = insertObjectMask(im(:,:,1), masks,Color=lines(size(masks, 3)) );
end

figure, imshow(overlayedImage)

% Show the bounding boxes and labels on the objects
%showShape("rectangle", gather(boxes), "Label", scores, "LineColor",'r')
toc


%%

