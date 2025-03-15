clear
clc
close all

load("4dnet.mat")

im=imread("A172_Phase_C7_1_00d00h00m_1.tif");
im=repmat(im, [1 1 4]);
%%
im=dlarray(single(im), 'SSCB');

Y=forward(dlnet, im);
maskSubnet = helper.extractMaskNetwork(dlnet);

%%
[boxes, scores, labels, masks] = detectMaskRCNN(dlnet, maskSubnet, im, params, "cpu");


%%
img=imread("A172_Phase_C7_1_00d00h00m_1.tif");

if(isempty(masks))
    overlayedImage = img;
else
    overlayedImage = insertObjectMask( img , masks,Color=lines(size(masks, 3)) );
end
figure, imshow(overlayedImage)


