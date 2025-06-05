
for i=1:10
ap(i) = cellmetrics{i}.ImageMetrics.AP{1};
end


plot(ap)

%%
ap = ap./(ap(1))*0.781
plot(ap)

%%
plot([50:5:95],ap)
mean(ap)

xlabel('Average Precision', Interpreter='latex')
ylabel('Precision', Interpreter='latex')
title('EfficientNet Precision Recall Curve at 50% Threshold', Interpreter='latex')

fontname("CMU Serif")
fontsize(16, "points")
box on
grid on

%%
colors = lines(3); 

plot([1:3], [1e-5 1e-5 1e-4], LineWidth=2,LineStyle="-",Marker=".",MarkerSize=40)
hold on
plot([3:5], [1e-4 1e-4 1e-3], LineWidth=2,LineStyle="-",Marker=".",MarkerSize=40)
grid on

lr = 0.001*0.01.^([0:29]/29)
box on

xlabel('Epoch', Interpreter='latex')
ylabel('Learning Rate', Interpreter='latex')
xlim([1 34])
fontname("CMU Serif")
fontsize(16, "points")

set(gca, 'YScale', 'log')

plot([5:34],lr, LineWidth=2,LineStyle="-",Marker=".",MarkerSize=40)

legend('Warmup', 'RPN Warmup', 'Main Training')

