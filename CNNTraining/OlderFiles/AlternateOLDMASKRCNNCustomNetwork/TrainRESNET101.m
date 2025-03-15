

clear
clc
close all

%Number of layers: 346
%Number of connections: 378

NumClasses=999; %number of categories
NumClasses=1+1; %image and background
ImSize= [520 704 3]; %image size

net=dlnetwork;

tempNet = [
    imageInputLayer([520 704 3],"Name","data")
    convolution2dLayer([7 7],64,"Name","conv1","BiasLearnRateFactor",0,"Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","bn_conv1")
    reluLayer("Name","conv1_relu")
    maxPooling2dLayer([3 3],"Name","pool1","Padding",[0 1 0 1],"Stride",[2 2])];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],64,"Name","res2a_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch2a")
    reluLayer("Name","res2a_branch2a_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2a_branch2b")
    reluLayer("Name","res2a_branch2b_relu")
    convolution2dLayer([1 1],256,"Name","res2a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res2a_branch1","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch1")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res2a")
    reluLayer("Name","res2a_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],64,"Name","res2b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2b_branch2a")
    reluLayer("Name","res2b_branch2a_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2b_branch2b")
    reluLayer("Name","res2b_branch2b_relu")
    convolution2dLayer([1 1],256,"Name","res2b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2b_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res2b")
    reluLayer("Name","res2b_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],64,"Name","res2c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2c_branch2a")
    reluLayer("Name","res2c_branch2a_relu")
    convolution2dLayer([3 3],64,"Name","res2c_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2c_branch2b")
    reluLayer("Name","res2c_branch2b_relu")
    convolution2dLayer([1 1],256,"Name","res2c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2c_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res2c")
    reluLayer("Name","res2c_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch2a")
    reluLayer("Name","res3a_branch2a_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn3a_branch2b")
    reluLayer("Name","res3a_branch2b_relu")
    convolution2dLayer([1 1],512,"Name","res3a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3a_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],512,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch1")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res3a")
    reluLayer("Name","res3a_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","res3b1_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b1_branch2a")
    reluLayer("Name","res3b1_branch2a_relu")
    convolution2dLayer([3 3],128,"Name","res3b1_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn3b1_branch2b")
    reluLayer("Name","res3b1_branch2b_relu")
    convolution2dLayer([1 1],512,"Name","res3b1_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b1_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res3b1")
    reluLayer("Name","res3b1_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","res3b2_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b2_branch2a")
    reluLayer("Name","res3b2_branch2a_relu")
    convolution2dLayer([3 3],128,"Name","res3b2_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn3b2_branch2b")
    reluLayer("Name","res3b2_branch2b_relu")
    convolution2dLayer([1 1],512,"Name","res3b2_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b2_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res3b2")
    reluLayer("Name","res3b2_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","res3b3_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b3_branch2a")
    reluLayer("Name","res3b3_branch2a_relu")
    convolution2dLayer([3 3],128,"Name","res3b3_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn3b3_branch2b")
    reluLayer("Name","res3b3_branch2b_relu")
    convolution2dLayer([1 1],512,"Name","res3b3_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b3_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res3b3")
    reluLayer("Name","res3b3_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch2a")
    reluLayer("Name","res4a_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4a_branch2b")
    reluLayer("Name","res4a_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4a_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],1024,"Name","res4a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch1")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4a")
    reluLayer("Name","res4a_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b1_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b1_branch2a")
    reluLayer("Name","res4b1_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b1_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b1_branch2b")
    reluLayer("Name","res4b1_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b1_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b1_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b1")
    reluLayer("Name","res4b1_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b2_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b2_branch2a")
    reluLayer("Name","res4b2_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b2_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b2_branch2b")
    reluLayer("Name","res4b2_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b2_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b2_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b2")
    reluLayer("Name","res4b2_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b3_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b3_branch2a")
    reluLayer("Name","res4b3_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b3_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b3_branch2b")
    reluLayer("Name","res4b3_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b3_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b3_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b3")
    reluLayer("Name","res4b3_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b4_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b4_branch2a")
    reluLayer("Name","res4b4_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b4_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b4_branch2b")
    reluLayer("Name","res4b4_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b4_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b4_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b4")
    reluLayer("Name","res4b4_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b5_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b5_branch2a")
    reluLayer("Name","res4b5_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b5_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b5_branch2b")
    reluLayer("Name","res4b5_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b5_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b5_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b5")
    reluLayer("Name","res4b5_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b6_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b6_branch2a")
    reluLayer("Name","res4b6_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b6_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b6_branch2b")
    reluLayer("Name","res4b6_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b6_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b6_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b6")
    reluLayer("Name","res4b6_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b7_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b7_branch2a")
    reluLayer("Name","res4b7_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b7_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b7_branch2b")
    reluLayer("Name","res4b7_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b7_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b7_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b7")
    reluLayer("Name","res4b7_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b8_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b8_branch2a")
    reluLayer("Name","res4b8_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b8_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b8_branch2b")
    reluLayer("Name","res4b8_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b8_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b8_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b8")
    reluLayer("Name","res4b8_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b9_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b9_branch2a")
    reluLayer("Name","res4b9_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b9_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b9_branch2b")
    reluLayer("Name","res4b9_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b9_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b9_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b9")
    reluLayer("Name","res4b9_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b10_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b10_branch2a")
    reluLayer("Name","res4b10_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b10_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b10_branch2b")
    reluLayer("Name","res4b10_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b10_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b10_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b10")
    reluLayer("Name","res4b10_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b11_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b11_branch2a")
    reluLayer("Name","res4b11_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b11_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b11_branch2b")
    reluLayer("Name","res4b11_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b11_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b11_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b11")
    reluLayer("Name","res4b11_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b12_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b12_branch2a")
    reluLayer("Name","res4b12_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b12_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b12_branch2b")
    reluLayer("Name","res4b12_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b12_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b12_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b12")
    reluLayer("Name","res4b12_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b13_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b13_branch2a")
    reluLayer("Name","res4b13_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b13_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b13_branch2b")
    reluLayer("Name","res4b13_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b13_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b13_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b13")
    reluLayer("Name","res4b13_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b14_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b14_branch2a")
    reluLayer("Name","res4b14_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b14_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b14_branch2b")
    reluLayer("Name","res4b14_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b14_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b14_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b14")
    reluLayer("Name","res4b14_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b15_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b15_branch2a")
    reluLayer("Name","res4b15_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b15_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b15_branch2b")
    reluLayer("Name","res4b15_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b15_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b15_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b15")
    reluLayer("Name","res4b15_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b16_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b16_branch2a")
    reluLayer("Name","res4b16_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b16_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b16_branch2b")
    reluLayer("Name","res4b16_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b16_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b16_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b16")
    reluLayer("Name","res4b16_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b17_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b17_branch2a")
    reluLayer("Name","res4b17_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b17_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b17_branch2b")
    reluLayer("Name","res4b17_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b17_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b17_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b17")
    reluLayer("Name","res4b17_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b18_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b18_branch2a")
    reluLayer("Name","res4b18_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b18_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b18_branch2b")
    reluLayer("Name","res4b18_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b18_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b18_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b18")
    reluLayer("Name","res4b18_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b19_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b19_branch2a")
    reluLayer("Name","res4b19_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b19_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b19_branch2b")
    reluLayer("Name","res4b19_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b19_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b19_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b19")
    reluLayer("Name","res4b19_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b20_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b20_branch2a")
    reluLayer("Name","res4b20_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b20_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b20_branch2b")
    reluLayer("Name","res4b20_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b20_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b20_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b20")
    reluLayer("Name","res4b20_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b21_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b21_branch2a")
    reluLayer("Name","res4b21_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b21_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b21_branch2b")
    reluLayer("Name","res4b21_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b21_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b21_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b21")
    reluLayer("Name","res4b21_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","res4b22_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b22_branch2a")
    reluLayer("Name","res4b22_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b22_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b22_branch2b")
    reluLayer("Name","res4b22_branch2b_relu")
    convolution2dLayer([1 1],1024,"Name","res4b22_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b22_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res4b22")
    reluLayer("Name","res4b22_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],512,"Name","res5a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn5a_branch2a")
    reluLayer("Name","res5a_branch2a_relu")
    convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn5a_branch2b")
    reluLayer("Name","res5a_branch2b_relu")
    convolution2dLayer([1 1],2048,"Name","res5a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5a_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],2048,"Name","res5a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn5a_branch1")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res5a")
    reluLayer("Name","res5a_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],512,"Name","res5b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5b_branch2a")
    reluLayer("Name","res5b_branch2a_relu")
    convolution2dLayer([3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn5b_branch2b")
    reluLayer("Name","res5b_branch2b_relu")
    convolution2dLayer([1 1],2048,"Name","res5b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5b_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res5b")
    reluLayer("Name","res5b_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],512,"Name","res5c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5c_branch2a")
    reluLayer("Name","res5c_branch2a_relu")
    convolution2dLayer([3 3],512,"Name","res5c_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn5c_branch2b")
    reluLayer("Name","res5c_branch2b_relu")
    convolution2dLayer([1 1],2048,"Name","res5c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5c_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","res5c")
    reluLayer("Name","res5c_relu")
    globalAveragePooling2dLayer("Name","pool5")
    fullyConnectedLayer(1000,"Name","fc1000")
    softmaxLayer("Name","prob")];
net = addLayers(net,tempNet);

% clean up helper variable
clear tempNet;

%Connect Layer Branches
%Connect all the branches of the network to create the network graph.
net = connectLayers(net,"pool1","res2a_branch2a");
net = connectLayers(net,"pool1","res2a_branch1");
net = connectLayers(net,"bn2a_branch2c","res2a/in1");
net = connectLayers(net,"bn2a_branch1","res2a/in2");
net = connectLayers(net,"res2a_relu","res2b_branch2a");
net = connectLayers(net,"res2a_relu","res2b/in2");
net = connectLayers(net,"bn2b_branch2c","res2b/in1");
net = connectLayers(net,"res2b_relu","res2c_branch2a");
net = connectLayers(net,"res2b_relu","res2c/in2");
net = connectLayers(net,"bn2c_branch2c","res2c/in1");
net = connectLayers(net,"res2c_relu","res3a_branch2a");
net = connectLayers(net,"res2c_relu","res3a_branch1");
net = connectLayers(net,"bn3a_branch2c","res3a/in1");
net = connectLayers(net,"bn3a_branch1","res3a/in2");
net = connectLayers(net,"res3a_relu","res3b1_branch2a");
net = connectLayers(net,"res3a_relu","res3b1/in2");
net = connectLayers(net,"bn3b1_branch2c","res3b1/in1");
net = connectLayers(net,"res3b1_relu","res3b2_branch2a");
net = connectLayers(net,"res3b1_relu","res3b2/in2");
net = connectLayers(net,"bn3b2_branch2c","res3b2/in1");
net = connectLayers(net,"res3b2_relu","res3b3_branch2a");
net = connectLayers(net,"res3b2_relu","res3b3/in2");
net = connectLayers(net,"bn3b3_branch2c","res3b3/in1");
net = connectLayers(net,"res3b3_relu","res4a_branch2a");
net = connectLayers(net,"res3b3_relu","res4a_branch1");
net = connectLayers(net,"bn4a_branch2c","res4a/in1");
net = connectLayers(net,"bn4a_branch1","res4a/in2");
net = connectLayers(net,"res4a_relu","res4b1_branch2a");
net = connectLayers(net,"res4a_relu","res4b1/in2");
net = connectLayers(net,"bn4b1_branch2c","res4b1/in1");
net = connectLayers(net,"res4b1_relu","res4b2_branch2a");
net = connectLayers(net,"res4b1_relu","res4b2/in2");
net = connectLayers(net,"bn4b2_branch2c","res4b2/in1");
net = connectLayers(net,"res4b2_relu","res4b3_branch2a");
net = connectLayers(net,"res4b2_relu","res4b3/in2");
net = connectLayers(net,"bn4b3_branch2c","res4b3/in1");
net = connectLayers(net,"res4b3_relu","res4b4_branch2a");
net = connectLayers(net,"res4b3_relu","res4b4/in2");
net = connectLayers(net,"bn4b4_branch2c","res4b4/in1");
net = connectLayers(net,"res4b4_relu","res4b5_branch2a");
net = connectLayers(net,"res4b4_relu","res4b5/in2");
net = connectLayers(net,"bn4b5_branch2c","res4b5/in1");
net = connectLayers(net,"res4b5_relu","res4b6_branch2a");
net = connectLayers(net,"res4b5_relu","res4b6/in2");
net = connectLayers(net,"bn4b6_branch2c","res4b6/in1");
net = connectLayers(net,"res4b6_relu","res4b7_branch2a");
net = connectLayers(net,"res4b6_relu","res4b7/in2");
net = connectLayers(net,"bn4b7_branch2c","res4b7/in1");
net = connectLayers(net,"res4b7_relu","res4b8_branch2a");
net = connectLayers(net,"res4b7_relu","res4b8/in2");
net = connectLayers(net,"bn4b8_branch2c","res4b8/in1");
net = connectLayers(net,"res4b8_relu","res4b9_branch2a");
net = connectLayers(net,"res4b8_relu","res4b9/in2");
net = connectLayers(net,"bn4b9_branch2c","res4b9/in1");
net = connectLayers(net,"res4b9_relu","res4b10_branch2a");
net = connectLayers(net,"res4b9_relu","res4b10/in2");
net = connectLayers(net,"bn4b10_branch2c","res4b10/in1");
net = connectLayers(net,"res4b10_relu","res4b11_branch2a");
net = connectLayers(net,"res4b10_relu","res4b11/in2");
net = connectLayers(net,"bn4b11_branch2c","res4b11/in1");
net = connectLayers(net,"res4b11_relu","res4b12_branch2a");
net = connectLayers(net,"res4b11_relu","res4b12/in2");
net = connectLayers(net,"bn4b12_branch2c","res4b12/in1");
net = connectLayers(net,"res4b12_relu","res4b13_branch2a");
net = connectLayers(net,"res4b12_relu","res4b13/in2");
net = connectLayers(net,"bn4b13_branch2c","res4b13/in1");
net = connectLayers(net,"res4b13_relu","res4b14_branch2a");
net = connectLayers(net,"res4b13_relu","res4b14/in2");
net = connectLayers(net,"bn4b14_branch2c","res4b14/in1");
net = connectLayers(net,"res4b14_relu","res4b15_branch2a");
net = connectLayers(net,"res4b14_relu","res4b15/in2");
net = connectLayers(net,"bn4b15_branch2c","res4b15/in1");
net = connectLayers(net,"res4b15_relu","res4b16_branch2a");
net = connectLayers(net,"res4b15_relu","res4b16/in2");
net = connectLayers(net,"bn4b16_branch2c","res4b16/in1");
net = connectLayers(net,"res4b16_relu","res4b17_branch2a");
net = connectLayers(net,"res4b16_relu","res4b17/in2");
net = connectLayers(net,"bn4b17_branch2c","res4b17/in1");
net = connectLayers(net,"res4b17_relu","res4b18_branch2a");
net = connectLayers(net,"res4b17_relu","res4b18/in2");
net = connectLayers(net,"bn4b18_branch2c","res4b18/in1");
net = connectLayers(net,"res4b18_relu","res4b19_branch2a");
net = connectLayers(net,"res4b18_relu","res4b19/in2");
net = connectLayers(net,"bn4b19_branch2c","res4b19/in1");
net = connectLayers(net,"res4b19_relu","res4b20_branch2a");
net = connectLayers(net,"res4b19_relu","res4b20/in2");
net = connectLayers(net,"bn4b20_branch2c","res4b20/in1");
net = connectLayers(net,"res4b20_relu","res4b21_branch2a");
net = connectLayers(net,"res4b20_relu","res4b21/in2");
net = connectLayers(net,"bn4b21_branch2c","res4b21/in1");
net = connectLayers(net,"res4b21_relu","res4b22_branch2a");
net = connectLayers(net,"res4b21_relu","res4b22/in2");
net = connectLayers(net,"bn4b22_branch2c","res4b22/in1");
net = connectLayers(net,"res4b22_relu","res5a_branch2a");
net = connectLayers(net,"res4b22_relu","res5a_branch1");
net = connectLayers(net,"bn5a_branch2c","res5a/in1");
net = connectLayers(net,"bn5a_branch1","res5a/in2");
net = connectLayers(net,"res5a_relu","res5b_branch2a");
net = connectLayers(net,"res5a_relu","res5b/in2");
net = connectLayers(net,"bn5b_branch2c","res5b/in1");
net = connectLayers(net,"res5b_relu","res5c_branch2a");
net = connectLayers(net,"res5b_relu","res5c/in2");
net = connectLayers(net,"bn5c_branch2c","res5c/in1");
net = initialize(net);


%%
Datdir="../";

ds = fileDatastore([Datdir+"DSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %training data
valds=fileDatastore([Datdir+"ValDSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %validation data
trainClassNames = ["CellA"];

%%
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.0005, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=1, ...
    LearnRateDropFactor=0.2, ...
    Plot="training-progress", ...
    Momentum=0.9, ...
    MaxEpochs=5, ...
    MiniBatchSize=6, ...
    ValidationData=valds, ...
    ValidationFrequency=10, ...
    BatchNormalizationStatistics="moving", ...
    ResetInputNormalization=false, ...
    ExecutionEnvironment="auto", ...
    VerboseFrequency=10, ...
    CheckpointFrequency=50, ...
    CheckpointFrequencyUnit='iteration') %, ...
   


  %%
  doTraining = true;
if doTraining
   % [net,info] = trainMaskRCNN(ds,net,options,FreezeSubNetwork="backbone");
    [net,info] = trainMaskRCNN(ds,net,options);
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save("trainedMaskRCNN-"+modelDateTime+".mat","net", "info");
end

