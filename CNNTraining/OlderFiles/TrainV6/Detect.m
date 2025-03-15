clear
clc
close all

addpath("src/")
% Read the image for inference
img = imread('A172_Phase_C7_1_00d00h00m_1.tif');
img=repmat(img, [1 1 3]);
load("Checkpoint--2024-11-25-20-03-45.mat")
net=savenet;
load("NET101.mat", "params")
maskSubnet = helper.extractMaskNetwork(net);

%%
% Define the target size of the image for inference
targetSize = [520 704 3];
trainCats = {'CellA'};

% Resize the image maintaining the aspect ratio and scaling the largest
% dimension to the target size.
imgSize = size(img);
[~, maxDim] = max(imgSize);
resizeSize = [NaN NaN];
resizeSize(maxDim) = targetSize(maxDim);

img = imresize(img, resizeSize);

% detect the objects and their masks
maskSubnet = helper.extractMaskNetwork(net);
classNames = trainCats;
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];
%params = createMaskRCNNConfig(imageSize, numClasses, classNames);
disp(params);
 
Threshold=params.Threshold

%%

params.NumRegionsToSample=2000;
params.NumStrongestRegions=5000;
params.RPNROIPerImage=600;
params.Threshold=0.4;


%%
params.Threshold=0.1;

img = imread('A172_Phase_C7_1_02d12h00m_2.tif');
img=repmat(img, [1 1 1]);

[boxes, scores, labels, masks] = detectMaskRCNN(net, maskSubnet, img, params, "cpu");

% Visualize Predictions

% Overlay the detected masks on the image using the insertObjectMask
% function.

%%
if(isempty(masks))
    overlayedImage = img;
else
    overlayedImage = insertObjectMask(img, masks,Color=lines(size(masks, 3)) );
end
figure, imshow(overlayedImage)

% Show the bounding boxes and labels on the objects
showShape("rectangle", gather(boxes), "Label", scores, "LineColor",'r')