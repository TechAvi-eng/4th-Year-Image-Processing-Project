clc
clear
close all

addpath("src/")
load("RESNET50.mat", "dlnet", "params")


%%
trainClassNames = ["CellA"];
imageSizeTrain = [520 704 3];

newnet = maskrcnn("resnet50-coco",trainClassNames,InputSize=imageSizeTrain)
%%
newnet.Layers=dlnet.Layers;
newnet.OutputNames={"rpnConv1x1ClsScores", 'rpnConv1x1BoxDeltas', 'rpl', 'bn5c_branch2c', 'rcnnSoftmax', 'mask_sigmoid1'}
