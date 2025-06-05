clear
clc
close all

addpath("src/")
% Read the image for inference
load("TestDSFs/label_A172_Phase_A7_1_00d00h00m_2.tif.mat")
load("DWTNet.mat", 'params')
load("CheckpointDWT.mat")



maskSubnet = helper.extractMaskNetwork(savenet);

%%
params.Threshold=0.0001;
params.OverlapThreshold=0.1;

[cA,cH,cV,cD] = dwt2(im,'sym4','mode','per');
im2=cat(3, cA,cH,cV,cD);


[boxes, scores, labels, masks] = detectMaskRCNN(savenet, maskSubnet, im2, params, "cpu");

% Visualize Predictions

% Overlay the detected masks on the image using the insertObjectMask
% % function.
%%
im3=im(1:2:end, 1:2:end);

%%

 



%%
if(isempty(masks))
    overlayedImage = im;
else
    overlayedImage = insertObjectMask(im3, masks,Color=lines(size(masks, 3)) );
end
figure, imshow(overlayedImage)

% Show the bounding boxes and labels on the objects
%showShape("rectangle", gather(boxes), "Label", scores, "LineColor",'r')