clear
clc
close all

load("trainedMaskRCNN-2024-11-12-00-45-27.mat") %load data from trained network

%%
%tiledlayout(2,1, "TileSpacing","tight")
%nexttile
ValLoss=vertcat(info.ValidationLoss);

%%
plot(vertcat(info.TrainingLoss))
hold on
set(gcf, "color", [1 1 1])

x=[1:numel(ValLoss)]';
xv=x(~isnan(ValLoss));
xlim([0 max(x)])

plot( x(~isnan(ValLoss)) , ValLoss( ~isnan(ValLoss) )   , LineWidth=2)

legend('Loss', 'Validation Loss', Interpreter='latex')
grid on

box on
xlabel('Iteration', Interpreter='latex')
ylabel('Loss', Interpreter='latex')
fontname("CMU Serif")
fontsize(16, "points")

%title("ResNet50: First Training Run", Interpreter='latex')
ylim([0 4])
%fontsize(24,"points")

%%
nexttile
plot(vertcat(info.TrainingMaskLoss))
hold on
ValLoss=vertcat(info.ValidationMaskLoss);
x=[1:numel(ValLoss)]';
xv=x(~isnan(ValLoss));
plot( x(~isnan(ValLoss)) , ValLoss( ~isnan(ValLoss) )   , LineWidth=2)
%legend('Loss', 'Validation Loss', Interpreter='latex')
grid on
box on
xlabel('Iteration', Interpreter='latex')
ylabel('Mask Loss', Interpreter='latex')
fontname("CMU Serif")
xlim([0 max(x)])


fontsize(16,"points")



%%
plot(vertcat(info.TrainingLoss))
hold on
set(gcf, "color", [1 1 1])

hold on
plot(movmean(vertcat(info.TrainingLoss), 100), LineWidth=3 )

legend('Loss', 'Validation Loss', Interpreter='latex')
grid on

box on
xlabel('Iteration', Interpreter='latex')
ylabel('Loss', Interpreter='latex')
fontname("CMU Serif")
fontsize(16, "points")

%title("ResNet50: First Training Run", Interpreter='latex')
ylim([0 4])
%fontsize(24,"points")