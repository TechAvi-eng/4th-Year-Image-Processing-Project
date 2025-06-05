clear
clc
close all

load('File2.mat', 'net');
net.OverlapThresholdRPN = 0.1;
net.OverlapThresholdPrediction =0.1;
%%
di = dir('./');

dirb = vertcat(di.bytes);

di(dirb==0, :)=[];

diname = {di.name};

for i = [1:length(diname)]

    load(strcat('./',diname{i} ) );

    numac(i) = size(masks, 3);
    
    confac(i) = sum(masks,"all")./(size(masks,1)*size(masks, 2));
    Thr = min(0.25*numac(i)*exp(-(numac(i)-250)./100 ), 0.25);
    
    im=rescale(im);
    [im, ~] = resizeImageandMask(im, [], [528, 704]);

    
    [masks,labels,scores,boxes] = segmentCells(net, im, "SegmentThreshold",Thr,"NumstrongestRegions",Inf, );
    numpred(i) = size(masks,3);
    i

end

%%
close all
scatter(numac, numpred, 32,'filled','ko')
xlabel('True Number of Cells', interpreter='latex')
ylabel('Predicted Number of Cells', interpreter='latex')
hold on
plot([0 1500], [0,1500], linestyle='--', linewidth=2, color=[1 1 1]*0.4 )
axis equal
xlim([0 1500])
ylim([0 1500])
box on
grid on
fontname('CMU Serif')
fontsize(16, 'points')
% set(gca, 'XScale', 'log')
% set(gca, 'YScale', 'log')
% xlim([10 2000])
% ylim([10 2000])


%%
MAPE = min(abs(numac-numpred)./numac*100, 100);
scatter(confac, MAPE, 32,'filled','ko')
ylabel('Difference in Count (\%)', interpreter='latex')
xlabel('Confluency', interpreter='latex')
%plot([0 1000], [0,1000], linestyle='--', linewidth=2, color=[1 1 1]*0.4 )
%axis equal
%xlim([0 1000])
%ylim([0 1000])
box on
grid on
fontname('CMU Serif')
fontsize(16, 'points')

%aa= fitlm(confac, MAPE);
%hold on
%plot(aa)
