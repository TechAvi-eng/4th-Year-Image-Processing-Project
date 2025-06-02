clear
clc
close all
addpath(genpath("~/Scratch/MRCNNsrc"));

load('NewestNet.mat', 'net');
net.OverlapThresholdRPN = 0.1;
net.OverlapThresholdPrediction =0.1;
%%
di = dir('./MidValDSFs');

dirb = vertcat(di.bytes);

di(dirb==0, :)=[];

diname = {di.name};
Thr =1e-5;

for i = [1:length(diname)]

    load(strcat('./MidValDSFs/',diname{i} ) );
    
    
    im=rescale(im);
    [im, ~] = resizeImageandMask(im, [], [528, 704]);
    
    
    [pmasks,plabels,pscores,pboxes] = segmentCells(net, im, "SegmentThreshold",Thr,"NumstrongestRegions",1000);
    
    allpboxes{i}=pboxes;
    allpsocres{i}=pscores;
    allabbox{i}=bbox;
    i
end

%%
[precision, recall, ap] = calculate_combined_pr_curve(allpboxes, allpsocres, allabbox, 0.075);

%%
for i = 1:569
numpred(i) = nnz(allpsocres{i}>0.2);
%numpred(i) = size(allpboxes{i},1);
numac(i) = size(allabbox{i},1);
end

close all
scatter(numac, numpred, 32,'filled','ko')
xlabel('True Number of Cells', interpreter='latex')
ylabel('Predicted Number of Cells', interpreter='latex')
hold on
plot([0 1500], [0,1500], linestyle='--', linewidth=2, color=[1 1 1]*0.4 )
axis equal
xlim([0 1200])
ylim([0 1200])
box on
grid on
fontname('CMU Serif')
fontsize(16, 'points')

%%
close all
plot([recall; 1],[precision; 0], LineWidth=3,Color='k')
xlabel('Recall', interpreter='latex')
ylabel('Precision', interpreter='latex')
title('Precision-Recall Curve at IoU=0.5')
hold on
axis equal
xlim([0 1])
ylim([0 1])
box on
grid on
fontname('CMU Serif')
fontsize(16, 'points')