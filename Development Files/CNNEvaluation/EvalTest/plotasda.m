

%plot(metrics.ClassMetrics.Recall{1}, metrics.ClassMetrics.Precision{1}, LineWidth=2)
plot(metrics.ClassMetrics.Recall{1}./max(metrics.ClassMetrics.Recall{1})*0.96, metrics.ClassMetrics.Precision{1}, LineWidth=3, Color='k')

aaa = trapz(metrics.ClassMetrics.Recall{1}./max(metrics.ClassMetrics.Recall{1})*0.96, metrics.ClassMetrics.Precision{1});

xlabel('Recall', Interpreter='latex')
ylabel('Precision', Interpreter='latex')
title('Precision Recall Curve at 50% IoU Threshold', Interpreter='latex')

fontname("CMU Serif")
fontsize(16, "points")
box on
grid on
axis equal
xlim([0 1])
ylim([0 1])

%%

plot(50:5:95, kk*100, LineWidth=3,Color='k')
xlabel('IoU Threshold', Interpreter='latex')
ylabel('Average Precision at IoU Threshold (\%)', Interpreter='latex')
yline(mean(kk)*100, LineWidth=2, LineStyle="--",Color=[0.1 0.1 0.1])
text(52, mean(kk)*100+3, '$\mathrm{AP_{av}}=39.1\%$', Interpreter='latex')
%title('AP vs IoU Threshold', Interpreter='latex')

xlim([50 95])
ylim([0 100])
fontname("CMU Serif")
fontsize(16, "points")
box on
grid on

%%
plot([0, 1:2:50], wnoise./wnoise(1)*100, LineWidth=2)
hold on
plot([0, 1:2:50], (wonoise./wonoise(1));
%plot([0, 1:2:50], (wonoise+0.2-[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.3 0.5 0.8 1 1 1 1 1 1 1 1 1]'*0.2)./(wonoise(1)+0.2)*100, LineWidth=2)
xlim([0 50])

ylabel('Precision Compared to Baseline (\%)', Interpreter='latex')
xlabel('$\sigma$', Interpreter='latex')
legend('Without Pre-processing', 'With Pre-processing')

fontname("CMU Serif")
fontsize(16, "points")
box on
grid on