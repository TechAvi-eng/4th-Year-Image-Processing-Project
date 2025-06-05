clear
clc
close all

%addpath("./vision") %when running on Myriad, needed to load the vision
%toolbox

ds = fileDatastore("./Synth/", ReadFcn=@(x)cocoAnnotationMATReader(x)); %training data
read(ds)

%%
data = preview(ds)
trainClassNames = ["CellA"];
imageSizeTrain = [520 704 3];

net = maskrcnn("resnet50-coco",trainClassNames,InputSize=imageSizeTrain)

%%
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.001, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=1, ...
    LearnRateDropFactor=0.8, ...
    Plot="none", ...
    Momentum=0.9, ...
    MaxEpochs=30, ...
    MiniBatchSize=1, ...
    BatchNormalizationStatistics="moving", ...
    ResetInputNormalization=false, ...
    ExecutionEnvironment="auto", ...
    VerboseFrequency=1);


%%

doTraining = true;
if doTraining
    [net,info] = trainMaskRCNN(ds,net,options) %,FreezeSubNetwork="backbone"); %to freeze a specific subnetwork from 
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save("trainedMaskRCNN-"+modelDateTime+".mat","net", "info");
end



%%
load("Synth/SynthBasicBox.mat")
%%

% %% utility to use model to test certain images
% im1=imread("../JSON_FORMATTING/LiveCellsIms1/livecell_test_images/A172_Phase_C7_1_00d00h00m_3.tif");
% 
     [masks,labels,scores,boxes] = segmentObjects(net,im,Threshold=0.5);
% 
% %%
% imshow(insertObjectMask(im1,masks, Color=lines(size(masks, 3))))

%%
if(isempty(masks))
    overlayedImage = im;
else
    overlayedImage = insertObjectMask(im, masks,Color=lines(size(masks, 3)) );
end
figure, imshow(overlayedImage)

% Show the bounding boxes and labels on the objects
showShape("rectangle", gather(boxes), "Label", scores, "LineColor",'r')