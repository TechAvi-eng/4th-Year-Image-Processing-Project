clear
clc
close all


datdir=".";


%ds = fileDatastore([datdir+"/DSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %training data

ds=fileDatastore("DSFs/label_A172_Phase_A7_1_00d00h00m_2.tif.mat", ReadFcn=@(x)cocoAnnotationMATReader(x))
%read(ds)
%valds=fileDatastore([datdir+"/ValDSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %validation data
%read(valds)

%%
%data = preview(ds)
trainClassNames = ["CellA"];
imageSizeTrain = [520 704 3];

%%
net = maskrcnn("resnet50-coco",trainClassNames,InputSize=imageSizeTrain)
%load("MaskRCNNResnet101.mat") %these are created using the createmaskRcnn function and (if using a network other than the default resnet 50) by modifying the matlab source code to customize each branch, see the following for the location in the MATLAB Source code: /Users/yigit/Documents/MATLAB/SupportPackages/R2024b/toolbox/vision/supportpackages/maskrcnn/data

schedule = timeBasedDecayLearnRate(0.01)


%%
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.02, ...
    LearnRateSchedule=schedule, ...
    Plot="none", ...
    Momentum=0.9, ...
    MaxEpochs=2, ...
    MiniBatchSize=1, ...
    BatchNormalizationStatistics="moving", ...
    ResetInputNormalization=false, ...
    ExecutionEnvironment="auto", ...
    VerboseFrequency=1, ...
    CheckpointFrequency=50, ...
    CheckpointFrequencyUnit='iteration', ...
    CheckpointPath="./") %training options
%% start training

doTraining = true;
if doTraining
   % [net,info] = trainMaskRCNN(ds,net,options,FreezeSubNetwork="backbone");
    [net,info] = trainMaskRCNN(ds,net,options);
    %modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
   % save("trainedMaskRCNN-"+modelDateTime+".mat","net", "info");
end


%% utility to use model to test certain images
im1=imread("../JSON_FORMATTING/LiveCellsIms1/livecell_test_images/A172_Phase_C7_1_00d00h00m_3.tif");

[masks,labels,scores,boxes] = segmentObjects(net,im1,Threshold=0.0001);
%%
imshow(insertObjectMask(im1,masks, Color=lines(size(masks, 3))))
