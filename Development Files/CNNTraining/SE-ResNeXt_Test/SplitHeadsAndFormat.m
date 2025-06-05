clear
clc
close all

addpath("/Users/yigit/Desktop/Stuff/Mechanical Engineering/4YP_Github/CNNTraining/SE-ResNeXt_Test")
[net net4 net5] = SEResNeXt101Backbone2([504 720, 1], 16, 8);


net4 = replaceLayer(net4,'stage4_block23_relu_final', reluLayer("Name",'feature_out'));


net5 = replaceLayer(net5, 'stage5_block3_relu_final', reluLayer('Name','postConvBackboneOut'));

s5l1=net5.Layers(1,1);
s5l1.Name='res5a_branch1';

s5l2=net5.Layers(14,1);
s5l2.Name='res5a_branch2a';

net5 = replaceLayer(net5, 'stage5_block1_conv1', s5l1);
net5 = replaceLayer(net5, 'stage5_block1_shortcut', s5l2);
net5 = removeLayers(net5, 'stage5_input');


%% save to file location to change next construction
dlnetFeature = net4;
dlnetPostFeature = net5;
clear net net4 net5 s5l1 s5l2 ans
save("/Users/yigit/Desktop/Stuff/Mechanical Engineering/4YP_Github/MRCNNsrc/NetData/ResNeXt101/dlnetFeature.mat","dlnetFeature")
save("/Users/yigit/Desktop/Stuff/Mechanical Engineering/4YP_Github/CNNTraining/BossaNova/NetDataResNext101/dlnetPostFeature.mat","dlnetPostFeature")

