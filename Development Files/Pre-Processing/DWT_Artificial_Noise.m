clear all
close all
clc

% Define folder and images
folder_path = 'test_images';
imageFiles = dir(fullfile(folder_path, '*.png'));
NumberImages = min(30, numel(imageFiles));   % Ensure we don't exceed available files

% Define noise levels (sigma in [0,40]) and denoising parameters
sigma_values = 0:5:50;                    % Standard deviations for Gaussian noise
displaySigma   = 10;                        % Sigma at which to display example images
numSigma     = numel(sigma_values);
waveletName  = 'db5';                      % Wavelet to use
level        = 4;                          % Decomposition level

% Pre-allocate result matrices
ssim_values = zeros(NumberImages, numSigma);
psnr_values = zeros(NumberImages, numSigma);

% Loop through images
for imageNumber = 1:NumberImages
    imageNumber
    % Get the full image path
    imagePath = fullfile(folder_path, imageFiles(imageNumber).name);
    % Read and convert to grayscale
    % Importing image
    im=imread(imagePath);

    % Coverts image to grayscale by taking mean of the 3 colour channels
    imgs = mean(im,3);
    imgs = rescale(imgs);

    % Loop through noise levels
    for sIdx = 1:numSigma
        sigma = sigma_values(sIdx);

        % Add Gaussian noise (scaled to [0,1])
        noise      = (sigma/255) * randn(size(imgs));
        img_noisy  = imgs + noise;
        img_noisy  = min(max(img_noisy, 0), 1);

        % Denoise using your function
        img_denoised = DWT_Denoise(img_noisy, 'Threshold', 0.1, ...
                                   'Level', level, 'Wavelet', waveletName);

        % If first image and specified sigma, display original, noisy, and denoised
        if imageNumber == 1 && sigma == displaySigma
            figure;
            t = tiledlayout(1,3,'Padding','None','TileSpacing','Compact');            
            nexttile(1)
            imshow(imgs);
            title('Original Image');
            nexttile(2)
            imshow(img_noisy);
            title(sprintf('Image with artificial\n Gaussian noise $\\sigma = %d$', sigma));
            nexttile(3)
            imshow(img_denoised);
            title('Denoised Image');
            t.Padding = 'compact';

        end

        % Compute metrics against the original clean image
        ssim_values(imageNumber, sIdx) = ssim(img_denoised, imgs);
        psnr_values(imageNumber, sIdx) = psnr(img_denoised, imgs);
    end


end

% Aggregate results: mean over all images
mean_ssim = mean(ssim_values, 1);
mean_psnr = mean(psnr_values, 1);

% Apply global settings for consistency
set(groot, 'DefaultTextInterpreter', 'latex');
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
set(groot, 'DefaultLegendInterpreter', 'latex');
set(groot, 'DefaultAxesFontName', 'CMU Serif');
set(groot, 'DefaultTextFontName', 'CMU Serif');

% Combined plot: PSNR (left) vs SSIM (right) across noise levels
figure;
yyaxis left;
plot(sigma_values, mean_psnr, '-x', 'LineWidth', 2.0);
ylabel('Mean PSNR (dB)');
hold on;
% Horizontal line at 90% of original (no-noise) PSNR
origPSNR = mean_psnr(1);
yline(0.9 * origPSNR, '--', '90% of original PSNR', 'LabelHorizontalAlignment', 'right', 'LineWidth', 1.0);
hold off;

yyaxis right;
plot(sigma_values, mean_ssim, '-x', 'LineWidth', 2.0);
ylabel('Mean SSIM');

xlabel('Gaussian Noise $\sigma$ (intensity)')

grid on;
grid minor;
