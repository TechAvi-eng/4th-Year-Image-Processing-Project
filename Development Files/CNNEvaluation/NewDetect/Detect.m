clear
clc
close all

addpath("src/")
% Read the image for inference
load("TestDSFs/label_A172_Phase_A7_1_00d00h00m_2.tif.mat")
load("1ChannelNet.mat", 'params')
load("Checkpoint1C.mat")



maskSubnet = helper.extractMaskNetwork(savenet);

%%

classNames = params.ClassNames;
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];
%params = createMaskRCNNConfig(imageSize, numClasses, classNames);
disp(params);
 
Threshold=params.Threshold;


%%

params.NumRegionsToSample=2000;
params.NumStrongestRegions=5000;
params.RPNROIPerImage=600;
params.Threshold=0.4;


%%
% params.Threshold=0.1;
% 
% im = imread('A172_Phase_C7_1_00d00h00m_1.tif');
% [cA,cH,cV,cD] = dwt2(im,'sym4','mode','per');
% im=cat(3, cA,cH,cV,cD);
% 
% 
% [boxes, scores, labels, masks] = detectMaskRCNN(net, maskSubnet, im, params, "cpu");
% 
% % Visualize Predictions
% 
% % Overlay the detected masks on the image using the insertObjectMask
% % function.

%%
clc
params.Threshold=0.1;

[boxes, scores, labels, masks] = detectMaskRCNN(savenet, maskSubnet, im, params, "cpu");
 



%%
if(isempty(masks))
    overlayedImage = im;
else
    overlayedImage = insertObjectMask(im(:,:), masks,Color=lines(size(masks, 3)) );
end
figure, imshow(overlayedImage)

% Show the bounding boxes and labels on the objects
showShape("rectangle", gather(boxes), "Label", labels, "LineColor",'r')