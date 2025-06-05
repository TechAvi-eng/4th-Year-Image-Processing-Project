clear all
close all
clc

% folder_path = 'test_images';
% imageFiles = dir(fullfile(folder_path, '*.png'));
% imagePath = fullfile(folder_path, imageFiles(1).name);
% 
% im=imread(imagePath);
% 
% imgs = mean(im,3);
% imgs = rescale(imgs);
% n = 1;
% wavelet_name = 'db1';
% reconstructed_img = DWT_Denoise(imgs, "Threshold", 0.1, "Level", n, "Wavelet", wavelet_name);
% 

% db_wavelets = {'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', ...
%                'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', ...
%                'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', ...
%                'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', ...
%                'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', ...
%                'db37', 'db38', 'db39', 'db40', 'db41', 'db42', 'db43', ...
%                'db44', 'db45'};
% 
% sym_wavelets = {'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', ...
%                 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', ...
%                 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20', ...
%                 'sym21', 'sym22', 'sym23', 'sym24', 'sym25', 'sym26', ...
%                 'sym27', 'sym28', 'sym29', 'sym30', 'sym31', 'sym32', ...
%                 'sym33', 'sym34', 'sym35', 'sym36', 'sym37', 'sym38', ...
%                 'sym39', 'sym40', 'sym41', 'sym42', 'sym43', 'sym44', ...
%                 'sym45'};
% 
% coif_wavelets = {'coif1', 'coif2', 'coif3', 'coif4', 'coif5'};
% 
% bior_wavelets = {'bior1.1', 'bior1.3', 'bior1.5', ...
%                  'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', ...
%                  'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', ...
%                  'bior4.4', 'bior5.5', 'bior6.8'};
% 
% rbio_wavelets = {'rbio1.1', 'rbio1.3', 'rbio1.5', ...
%                  'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', ...
%                  'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', ...
%                  'rbio4.4', 'rbio5.5', 'rbio6.8'};

% db_wavelets = {'db1', 'db3', 'db5', 'db7', ...
%                'db9', 'db11', 'db13', 'db15', ...
%                'db17', 'db19',};
% 
% sym_wavelets = {'sym2', 'sym4', 'sym6', 'sym8', ...
%                 'sym10', 'sym12', 'sym14', ...
%                 'sym16', 'sym18', 'sym20'};
% 
% coif_wavelets = {'coif1', 'coif3', 'coif5'};
% 
% bior_wavelets = {'bior1.1', 'bior1.5', ...
%                  'bior2.2', 'bior2.6', ...
%                  'bior3.1', 'bior3.5', ...
%                  'bior4.4', 'bior5.5', 'bior6.8'};
% 
% rbio_wavelets = {'rbio1.1', 'rbio1.5', ...
%                  'rbio2.2', 'rbio2.6', ...
%                  'rbio3.1', 'rbio3.5', ...
%                  'rbio4.4', 'rbio5.5', 'rbio6.8'};

db_wavelets = {'db1', 'db10', 'db20',};

sym_wavelets = {'sym2', 'sym10', 'sym20',};

coif_wavelets = {'coif1','coif5'};

bior_wavelets = {'bior1.1','bior4.4', 'bior6.8'};

rbio_wavelets = {'rbio1.1','rbio4.4', 'rbio6.8'};
wavelets = [db_wavelets, sym_wavelets, coif_wavelets, bior_wavelets, rbio_wavelets];


% Define folder containing test images
folder_path = 'test_images';
imageFiles = dir(fullfile(folder_path, '*.png'));


NumberImages = 20;
% wavelets = db_wavelets;
n = 5;

ssim_values = zeros(NumberImages, length(wavelets), n);
psnr_values = zeros(NumberImages, length(wavelets), n);
elapsed_times = zeros(NumberImages, length(wavelets));


for imageNumber = 1:NumberImages
    imageNumber
    % Get the full image path
    imagePath = fullfile(folder_path, imageFiles(imageNumber).name);

    % Importing image
    im=imread(imagePath);

    % Coverts image to grayscale by taking mean of the 3 colour channels
    imgs = mean(im,3);
    imgs = rescale(imgs);


    % Loop through each wavelet function
    for w = 1:length(wavelets)
        wavelet_name = wavelets{w};

        for level = 1:n
            % Timing start
            tic;
            % Apply wavelet function here
            reconstructed_img = DWT_Denoise(imgs, "Threshold", 0.1, "Level", level, "Wavelet", wavelet_name);

            % Timing end and saving time
            elapsed_times(imageNumber, w) = toc;

            % Calculate SSIM between the original and reconstructed image
            ssim_val = ssim(reconstructed_img, imgs);

            % Calculate PSNR between the original and reconstructed image
            peaksnr_val = psnr(reconstructed_img, imgs);

            ssim_values(imageNumber, w, level) = ssim_val; % Store the SSIM value
            psnr_values(imageNumber, w, level) = peaksnr_val; % Store the PSNR value

        end

    end

end

% Calculating average metrics across all images
average_ssim = mean(ssim_values, 1);
average_ssim = squeeze(average_ssim);
average_psnr = mean(psnr_values, 1);
average_psnr = squeeze(average_psnr);
average_elapsed_time = mean(elapsed_times, 1);

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

grid on;
grid minor;
hold off;
