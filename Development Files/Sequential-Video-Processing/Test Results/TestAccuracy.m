clear
clc
close all

load('Res101-FINAL.mat', 'net');


%%
clear im
imA(:,:,1) = imread("VID1282_C6_1_00d00h00m.tif");
imA(:,:,2) = imread("VID1282_C6_1_00d02h00m.tif");
imA(:,:,3) = imread("VID1282_C6_1_00d04h00m.tif");
imA(:,:,4) = imread("VID1282_C6_1_00d06h00m.tif");
imA(:,:,5) = imread("VID1282_C6_1_00d08h00m.tif");
imA(:,:,6) = imread("VID1282_C6_1_00d10h00m.tif");
imA(:,:,7) = imread("VID1282_C6_1_00d12h00m.tif");


s = 400;
S = 300;

for i=1:7

imB(:,:,i) = imA(s+1:s+528/2, S+1:S+704/2, i);
im(:,:,i)= imresize(imB(:,:,i), 2);
end
clear imA imB


%%

net.OverlapThreshold=0.2;
[Masks Cats Scores Boxes] = segmentCells(net, im(:,:,2), "ShowMasks",true, NumstrongestRegions=Inf, SegmentThreshold=0.1, ShowScores=false);

%%
tic
[Masks Cats Scores Boxes RPN1] = segmentFrame(net, im(:,:,1), [], "NumStrongestRegions",Inf, Threshold=0.1);
toc

overlayedImage = insertObjectMask(im(:,:,1), Masks,Color=lines(size(Masks, 3)) );
figure, imshow(overlayedImage)

%%
clear Time Num
Time=[];
Num=[];

for alp=[0:0.01:1]
tic
[Masks Cats Scores Boxes RPN] = segmentFrame(net, im(:,:,2), RPN1, "NumStrongestRegions",Inf, Threshold=0.1,Alpha=alp);
Time(end+1)=toc;
Num(end+1)=size(Masks,3);
end



%%
alp=[0:0.01:1];

tiledlayout(2,1,"TileSpacing","compact")

nexttile

plot(alp, Num/152*100)
ylabel('Recall (\%)', interpreter='latex')
xline(0.15, 'r--')
grid on

nexttile

plot(alp, Time)
xline(0.15, 'r--')
grid on

ylabel('Average Processing Time (s)', interpreter='latex')
fontname('CMU Serif')
fontsize(16, "points")
xlabel('$\alpha$', interpreter='latex', FontSize=24)

%%
overlayedImage = insertObjectMask(im(:,:,2), Masks,Color=lines(size(Masks, 3)) );
figure, imshow(overlayedImage)



%%
net.ProposalsOutsideImage='clip';
     [masks,labels,scores,boxes] = segmentFrameviaKnowns(net,im,boxes,Threshold=0.5,NumStrongestRegions=1200, NumAdditionalProposals=2, SelectStrongest=true, MinSize=[8 8],MaxSize=[80 80]);
%%

[masks,labels,scores,boxes1] = segmentFrameviaKnowns(net, im(:,:,1), [], threshold=0.1);
[masks,labels,scores,boxes,RPN1] = segmentFrame(net, im(:,:,1), [], "NumStrongestRegions",Inf, Threshold=0.1);


%%
num1 = [];
num2 = [];
numA = [];

for i=2:7

[masks,labels,scores,boxes] = segmentFrameviaKnowns(net, im(:,:,i), [],NumAdditionalProposals=10000, threshold=0.1);
numA(end+1)=size(masks,3);

[masks,labels,scores,boxes2] = segmentFrameviaKnowns(net, im(:,:,i), boxes1,NumAdditionalProposals=100, threshold=0.1);

num1(end+1)=size(masks,3);

[masks,labels,scores,boxes,RPN1] = segmentFrame(net, im(:,:,i), RPN1, "NumStrongestRegions",Inf, Threshold=0.1);

num2(end+1)=size(masks,3);

end


plot(numA)
hold on

plot(num1)
plot(num2)

%%
t=[2:2:12];
plot(t,abs((num1-numA))./numA*100, LineWidth=2)
hold on
plot(t,abs((num2-numA))./numA*100, LineWidth=2)

xlabel('Time between Frames (hrs)', interpreter='latex', FontSize=24)
ylabel('Absolute Error from Independent Predictions (\%)', interpreter='latex', FontSize=24)

legend('Reuse Boxes as Proposals', 'Modified RPN', interpreter='latex', location='NorthWest')

grid on
box on

fontname('CMU Serif')
fontsize(16, "points")

%%
tiledlayout(1,3, "TileSpacing","compact","Padding","tight")
nexttile
imshow(im(:,:,1))
title('Initial State', interpreter='latex')
nexttile
imshow(im(:,:,2))
title('2 Hours', interpreter='latex')

nexttile
imshow(im(:,:,5))
title('10 Hours', interpreter='latex')

fontname('CMU Serif')
fontsize(16, "points")
