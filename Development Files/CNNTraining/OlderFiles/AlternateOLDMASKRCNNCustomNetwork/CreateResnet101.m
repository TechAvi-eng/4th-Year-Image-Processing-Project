

clear
clc
close all

addpath(genpath("./src"))

imageSize = [520 704 3];

trainCats={'CellA'};
classNames = trainCats;
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];

params = createMaskRCNNConfig(imageSize, numClasses, classNames);
disp(params)

%%
net=createMaskRCNN(numClasses,params,"resnet101")
