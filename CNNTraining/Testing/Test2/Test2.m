clear
clc
close all

%%
ds = fileDatastore("./SingleDS/", ReadFcn=@(x)MATReader1C(x, 0)); %training data
data = preview(ds)


%%
trainClassNames = ["CellA"];
imageSizeTrain = [528 704 1];

ABs = [14 14; 14 21; 21 14;...
    21 21; 21 32; 32 21;...
    32 32; 47 32; 32 47;...
    47 47; 71 47; 47 71;...
    71 71];

net = MRCNN(trainClassNames,ABs,InputSize=imageSizeTrain, ScaleFactor=[1 1]/16,ModelName='ResNet50')

%%

options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=125, ...
    LearnRateDropFactor=0.5, ...
    Plot="none", ...  
    MaxEpochs=200, ...
    MiniBatchSize=1, ...
    ResetInputNormalization=false, ...
    ExecutionEnvironment="cpu", ...
    VerboseFrequency=1, ...
    L2Regularization=1e-5) 

%%
[net,info] = trainMRCNN(ds,net,options, NumStrongestRegions=1000, NumRegionsToSample=100, PositiveOverlapRange=[0.5 1], NegativeOverlapRange=[0 0.5], ForcedPositiveProposals=false, FreezeSubNetwork="none")



%%
if ~exist("im")
    load("SingleDS/label_A172_Phase_A7_1_00d00h00m_2.tif.mat", "im")
    im=rescale(im);
    im=repmat(im ,[1 1 1]); 
end


tic
for i = 1
[masks,labels,scores,boxes] = segmentObjects(net,im,Threshold=0.5,NumStrongestRegions=1000, SelectStrongest=true, MinSize=[4 4],MaxSize=[80 80] );

if(isempty(masks))
    overlayedImage = im(:,:,1);
else
    overlayedImage = insertObjectMask(im(:,:,1), masks,Color=lines(size(masks, 3)) );
end
%%
figure, imshow(overlayedImage)
showShape("rectangle", gather(boxes), "Label", scores, "LineColor",'r')

end
toc


%%
if ~exist("im")
    load("SingleDS/label_A172_Phase_A7_1_00d00h00m_2.tif.mat", "im")
    im=rescale(im);
    im=repmat(im ,[1 1 1]); 
end


dlX = dlarray(im, 'SSCB');
dlFeatures = predict(net.FeatureExtractionNet, dlX, 'Acceleration','auto');
    
[dlRPNScores, dlRPNReg] = predict(net.RegionProposalNet, dlFeatures, 'Outputs',{'RPNClassOut', 'RPNRegOut'});

%%
for i = 1:1024

    d = extractdata(dlFeatures(:,:,i));
    if max(d, [], "all")>0.1
    
    imagesc(d); colorbar
    title(num2str(i))
    %set(gca,'ColorScale','log')

        pause(0.2)
    end

end

%%
[see] = predict(net.RegionProposalNet, dlFeatures, 'Outputs',{'RPNClassOut'});

%%

for i = 1:13

    d = (extractdata(see(:,:,i)));
    if 1
    
    imagesc(d); colorbar
    title(num2str(i))
    %set(gca,'ColorScale','log')
    %caxis([0 1])
        pause(0.6)
    end

end

