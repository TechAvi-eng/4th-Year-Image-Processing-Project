clear all
close all
clc

load("Performance_Results_Condensed.mat")
load("Wavelets.mat")

% Apply global settings for consistency
set(groot, 'DefaultTextInterpreter', 'latex');
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
set(groot, 'DefaultLegendInterpreter', 'latex');
set(groot, 'DefaultAxesFontName', 'CMU Serif');
set(groot, 'DefaultTextFontName', 'CMU Serif');

% --- SSIM Plot ---
figure;
t = tiledlayout(1,3,'Padding','None','TileSpacing','Compact');            
nexttile(1)
plot(average_ssim(:,1), 'x', 'LineWidth', 2); hold on;
plot(average_ssim(:,2), 'x', 'LineWidth', 2);
plot(average_ssim(:,3), 'x', 'LineWidth', 2);
plot(average_ssim(:,4), 'x', 'LineWidth', 2);
plot(average_ssim(:,5), 'x', 'LineWidth', 2);

num_wavelets = size(average_ssim, 1);
set(gca, 'XTick', 1:num_wavelets);
set(gca, 'XTickLabel', wavelets);

xlabel('Wavelet Function');
ylabel('SSIM');
title('SSIM Values');
legend({'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'}, 'Location', 'best');
t.Padding = 'compact';
xlim([1 num_wavelets]);

grid on;
grid minor;
hold off;


% --- PSNR Plot ---
nexttile(2)
plot(average_psnr(:,1), 'x', 'LineWidth', 2); hold on;
plot(average_psnr(:,2), 'x', 'LineWidth', 2);
plot(average_psnr(:,3), 'x', 'LineWidth', 2);
plot(average_psnr(:,4), 'x', 'LineWidth', 2);
plot(average_psnr(:,5), 'x', 'LineWidth', 2);

num_wavelets = size(average_psnr, 1);
set(gca, 'XTick', 1:num_wavelets);
set(gca, 'XTickLabel', wavelets);

xlabel('Wavelet Function');
ylabel('PSNR (dB)');
title('PSNR Values');
legend({'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'}, 'Location', 'best');
t.Padding = 'compact';
xlim([1 num_wavelets]);

grid on;
grid minor;
hold off;


% --- Elapsed Time Plot ---
nexttile(3)
plot(average_elapsed_time, 'x', 'LineWidth', 2); hold on;

num_wavelets = size(average_psnr, 1);
set(gca, 'XTick', 1:num_wavelets);
set(gca, 'XTickLabel', wavelets);

xlabel('Wavelet Function');
ylabel('Time (s)');
title('Time for Computations');
t.Padding = 'compact';
xlim([1 num_wavelets]);

grid on;
grid minor;
hold off;
