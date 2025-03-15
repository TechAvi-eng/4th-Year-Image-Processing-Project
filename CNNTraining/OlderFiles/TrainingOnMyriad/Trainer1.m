clear
clc
close all


%datdir="/home/zcemydo/Scratch/TrainV2/";
datdir="..";

ds = fileDatastore([datdir+"/SmallDSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %training data
%read(ds)
%%
%valds=fileDatastore([datdir+"/ValDSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %validation data
%read(valds)

%%

rng(1863);
ds=shuffle(ds); %randomize order of the dataset

data = preview(ds)

trainClassNames = ["CellA"];
imageSizeTrain = [520 704 4];

%%
%net = maskrcnn("resnet50-coco",trainClassNames,InputSize=imageSizeTrain)
load("MaskRCNNResnet50.mat") %these are created using the createmaskRcnn function and (if using a network other than the default resnet 50) by modifying the matlab source code to customize each branch, see the following for the location in the MATLAB Source code: /Users/yigit/Documents/MATLAB/SupportPackages/R2024b/toolbox/vision/supportpackages/maskrcnn/data

availableGPUs = gpuDeviceCount("available") %start parallel process if GPUs are available
if availableGPUs>0
  parpool("Processes",availableGPUs)
end



%%
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.001, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=1, ...
    LearnRateDropFactor=0.1, ...
    Plot="none", ...
    Momentum=0.9, ...
    MaxEpochs=3, ...
    MiniBatchSize=1, ...
    ValidationData=[], ...
    ValidationFrequency=10, ...
    BatchNormalizationStatistics="moving", ...
    ResetInputNormalization=false, ...
    ExecutionEnvironment="auto", ...
    VerboseFrequency=1, ...
    CheckpointFrequency=40, ...
    CheckpointFrequencyUnit='iteration', ...
    CheckpointPath="./") %training options

%% start training

doTraining = true;
if doTraining
   % [net,info] = trainMaskRCNN(ds,net,options,FreezeSubNetwork="backbone");
    [net,info] = trainMaskRCNN(ds,net,options, ...
        NumStrongestRegions=Inf)

    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save("trainedMaskRCNN-"+modelDateTime+".mat","net", "info"); %save output with the date and time into the current directotry
end



% %% utility to use model to test certain images
% im1=imread("../JSON_FORMATTING/LiveCellsIms1/livecell_test_images/A172_Phase_C7_1_00d00h00m_3.tif");
%
% [masks,labels,scores,boxes] = segmentObjects(net,im1,Threshold=0.5);
%
% %%
% imshow(insertObjectMask(im1,masks, Color=lines(size(masks, 3))))
