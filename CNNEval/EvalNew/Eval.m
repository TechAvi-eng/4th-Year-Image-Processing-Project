clear
clc
close all

load('File2.mat', 'net');
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
    
    
    [pmasks,plabels,pscores,pboxes] = segmentCells(net, im, "SegmentThreshold",Thr,"NumstrongestRegions",Inf);
    
    allpboxes{i}=pboxes;
    allpsocres{i}=pscores;
    allabbox{i}=bbox;
    i
end

calculate_combined_pr_curve(allpboxes, allpsocres, allabbox, 0.5)