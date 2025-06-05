clear all
close all
clc

load("Performance_Metrics_New.mat")
load("Wavelets.mat")

% Apply global settings for consistency
set(groot, 'DefaultTextInterpreter', 'latex');
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
set(groot, 'DefaultLegendInterpreter', 'latex');
set(groot, 'DefaultAxesFontName', 'CMU Serif');
set(groot, 'DefaultTextFontName', 'CMU Serif');

% --- SSIM Plot ---
figure;
plot(average_ssim(:,1), 'x', 'LineWidth', 2); hold on;
plot(average_ssim(:,2), 'x', 'LineWidth', 2);
plot(average_ssim(:,3), 'x', 'LineWidth', 2);
plot(average_ssim(:,4), 'x', 'LineWidth', 2);
plot(average_ssim(:,5), 'x', 'LineWidth', 2);

num_wavelets = size(average_ssim, 1);
set(gca, 'XTick', 1:num_wavelets);
set(gca, 'XTickLabel', wavelets);

xlabel('Wavelet Function');
ylabel('SSIM (Structural Similarity Index)');
title('SSIM Values against Wavelet Functions');
legend({'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'}, 'Location', 'best');
t.Padding = 'compact';
xlim([1 num_wavelets]);

grid on;
grid minor;
hold off;


% --- PSNR Plot ---
figure;
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
title('PSNR Values against Wavelet Functions');
legend({'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'}, 'Location', 'best');
t.Padding = 'compact';
xlim([1 num_wavelets]);

grid on;
grid minor;
hold off;


% --- Elapsed Time Plot ---
figure;
plot(average_elapsed_time, 'x', 'LineWidth', 2); hold on;

num_wavelets = size(average_psnr, 1);
set(gca, 'XTick', 1:num_wavelets);
set(gca, 'XTickLabel', wavelets);

xlabel('Wavelet Function');
ylabel('Time (s)');
title('Time for Discrete Wavelet Transform Computations');
t.Padding = 'compact';
xlim([1 num_wavelets]);

grid on;
grid minor;
hold off;
