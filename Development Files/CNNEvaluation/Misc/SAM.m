clear
clc
close all

im=imread("TestIms/");


[masks,scores] = imsegsam(im);

%%
labelMatrix = labelmatrix(masks);
maskOverlay = labeloverlay(im,labelMatrix);
imshow(maskOverlay,[])