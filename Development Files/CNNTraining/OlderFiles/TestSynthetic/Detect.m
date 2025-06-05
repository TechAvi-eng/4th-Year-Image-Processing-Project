

addpath("src/")
load("newcircs/circs.mat")


maskSubnet = helper.extractMaskNetwork(dlnet);


%%

params.Threshold=0.000001;


[boxes, scores, labels, masks] = detectMaskRCNN(dlnet, maskSubnet, im, params, "cpu");

% Visualize Predictions

% Overlay the detected masks on the image using the insertObjectMask
% function.


if(isempty(masks))
    overlayedImage = im;
else
    overlayedImage = insertObjectMask(im, masks,Color=lines(size(masks, 3)) );
end
figure, imshow(overlayedImage)

% Show the bounding boxes and labels on the objects
showShape("rectangle", gather(boxes), "Label", scores, "LineColor",'r')