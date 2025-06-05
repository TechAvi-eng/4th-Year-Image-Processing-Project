a =MatlabImportToolpaste13882179130443424212tmp;

%%
a = a(~isnan(a(:,1)),:);
%%
a=horzcat([1:22340]', a);
%%
close all
plot(a(:,1)./(length(a)/34), a(:,2))
hold on
plot(a(:,1)./(length(a)/34), movmean(a(:,2),250), LineWidth=3)
%vald = ~isnan(a(:,1));

%plot(a.VarName2(vald)./3188*2, a.VarName11(vald)-0.5, LineWidth=3)                                     %plot([4:2:34], [6.871, 4.3123, 2.843, 1.920174, 1.763, 1.6695, 1.732, 1.602981, 1.592483, 1.5234890, 1.452, 1.396 1.37 1.2023 1.19745, 1.18], LineWidth=3)
xregion(0,4)

legend('Training Loss', '250 Iterations Moving Average', 'Validation Loss', 'Warmup period', Interpreter='latex')
xlabel('Epoch', Interpreter='latex')
ylabel('Loss', Interpreter='latex')
fontname('CMU Serif')
fontsize(16, 'points')
xlim([0 34])
ylim([0 3])

box on
grid on

%%

