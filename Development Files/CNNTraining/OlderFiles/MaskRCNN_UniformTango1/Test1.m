clear
clc
close all

%%
addpath(genpath("./vision")) %when running on Myriad, needed to load the vision
%toolbox

ds = fileDatastore("./Pls/", ReadFcn=@(x)MATReader1C(x)); %training data
read(ds)
data = preview(ds)


%%
trainClassNames = ["CellA"];
imageSizeTrain = [520 704 1];

net = maskrcnn("resnet50-coco",trainClassNames,InputSize=imageSizeTrain)

%%
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.005, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=1, ...
    LearnRateDropFactor=1, ...
    Plot="none", ...
    Momentum=0.9, ...
    MaxEpochs=100, ...
    MiniBatchSize=1, ...
    BatchNormalizationStatistics="moving", ...
    ResetInputNormalization=false, ...
    ExecutionEnvironment="auto", ...
    VerboseFrequency=1);


%%

doTraining = true;
if doTraining
    [net,info] = trainMaskRCNN(ds,net,options,"NegativeOverlapRange",[0.1 0.5]) 
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save("trainedMaskRCNN-"+modelDateTime+".mat","net", "info");
end



%%
load("Pls/TestCircs.mat", "im")
im=uint8(rescale(im,0, 255));
%%

% %% utility to use model to te
% st certain images
% im1=imread("../JSON_FORMATTING/LiveCellsIms1/livecell_test_images/A172_Phase_C7_1_00d00h00m_3.tif");
% 
     [masks,labels,scores,boxes] = segmentObjects(net,im,Threshold=0.1);
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
showShape("rectangle", gather(boxes), "Label", scores, "LineColor",'r')