clear
clc
close all

%load("trainedMaskRCNN-2024-11-12-00-45-27.mat")


trainClassNames = ["CellA"];
imageSizeTrain = [520 704 3];

net = maskrcnn("resnet50-coco",trainClassNames,InputSize=imageSizeTrain)



%%
%im=imread("TestIms/A172_Phase_C7_1_00d00h00m_1.tif");
im=imread("A172_Phase_C7_1_00d00h00m_1.tif");

im=repmat(im, [1 1 3]);


%%
[masks,labels,scores,boxes] = segmentObjects(newnet,im,Threshold=0.5);



overlayedImage = insertObjectMask(im,masks);
imshow(overlayedImage);
showShape("rectangle",boxes,Label=labels,LineColor=[1 0 0]);
