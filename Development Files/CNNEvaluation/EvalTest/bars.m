
Res = [78.1 70.9 62.42];
x= {'EfficientNet', 'ResNet101', 'ResNet50'};

x=categorical(x)
bar(x, Res)
ylim([0 100])

errhigh = [92.1, 85.4, 70.2] - Res;

errlow = Res - [45 42 30];
hold on


er = errorbar(x, Res,errlow,errhigh);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
er.LineWidth = 2;


%xlabel('Backbone', Interpreter='latex')
ylabel('$\mathrm{AP_{50}}$ (\%)', Interpreter='latex')
title('Precision for Backbones', Interpreter='latex')
xlim([x(1) x(3)])
fontname("CMU Serif")
fontsize(16, "points")
box on
grid on