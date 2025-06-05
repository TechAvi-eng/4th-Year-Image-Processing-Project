
clear
clc
close all


addpath('./src');

%%


%trainClassNames = ["CellA"];
imageSize=[520 704 4];

trainCats = {'CellA'};


classNames = trainCats;
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];


 datdir="..";
 ds = fileDatastore([datdir+"/SmallDSFs"], ReadFcn=@(x)cocoAnnotationMATReader4(x)); %training datatrainDS = transform(ds, @(x)helper.preprocessData(x, imageSize));


trainDS = ds.shuffle;


data = preview(trainDS) 






%dlnet = createMaskRCNN(numClasses, params, 'resnet50'); %or resnet 101
load("4dnet.mat")
%load("NET101.mat")


%set environment
if canUseGPU
    executionEnvironment = "gpu";
    gpuDevice(1)
else
    executionEnvironment = "cpu";
end

params.ImageSize=[520 704 3];
%disp(params);

RPNRegDeltas = {'rpnConv1x1BoxDeltas'};regionProposal = {'rpl'};
 
outputNodes = [ RPNRegDeltas, regionProposal, dlnet.OutputNames(:)'];


%%

X=dlarray(single(repmat(imread("A172_Phase_C7_1_00d00h00m_1.tif"), [1 1 4])), 'SSCB');

%%

[YRPNRegDeltas, proposal, YRCNNClass, YRCNNReg, YRPNClass, YMask, state] = forward(...
                                            dlnet, X, 'Outputs', outputNodes);


%%
size(YRCNNClass)

Y=extractdata(YRCNNClass);