clear all
close all
clc

%%

% Define folder containing test images
folder_path = 'test_images';
imageFiles = dir(fullfile(folder_path, '*.png'));

% Set up wavelet parameters
waveletType = 'haar'; % Change this to the desired wavelet type
numLevels = 2; % Number of decomposition levels
threshold_max = 200; % Maximum threshold to iterate through

for imageNumber = 1:1
    % Get the full image path
    imagePath = fullfile(folder_path, imageFiles(imageNumber).name);

    % Importing image
    im = imread(imagePath);
    
    % Convert image to grayscale if it's in color
    if size(im, 3) == 3
        imgs = mean(im, 3);
    else
        imgs = im;
    end
    
    imgs = double(imgs); % Convert to double for processing
    
    % Initialize arrays to store metrics
    threshold_vals = 1:threshold_max;
    err = zeros(1, threshold_max);
    peaksnr = zeros(1, threshold_max);
    ssimval = zeros(1, threshold_max);
    
    for threshold = threshold_vals
        % Perform the DWT decomposition
        [c, s] = wavedec2(imgs, numLevels, waveletType);
        
        % Apply thresholding to the wavelet coefficients
        c_denoised = c;
        c_denoised(abs(c_denoised) < threshold) = 0;
        
        % Reconstruct the image from thresholded coefficients
        img_filtered = waverec2(c_denoised, s, waveletType);
        
        % Rescale images for comparison
        imgs_rescaled = rescale(imgs);
        img_filtered_rescaled = rescale(img_filtered);
        
        % Calculate error metrics
        err(threshold) = immse(img_filtered_rescaled, imgs_rescaled);
        peaksnr(threshold) = psnr(img_filtered_rescaled, imgs_rescaled);
        ssimval(threshold) = ssim(img_filtered_rescaled, imgs_rescaled);
    end
    
    % Store the metrics for the current image
    err_all(imageNumber, :) = err;
    peaksnr_all(imageNumber, :) = peaksnr;
    ssimval_all(imageNumber, :) = ssimval;
    
    disp(['Processed image ', num2str(imageNumber), ' of ', num2str(length(imageFiles))]);
end

% Calculate mean of statistics across all images
err_mean = mean(err_all, 1);
peaksnr_mean = mean(peaksnr_all, 1);
ssimval_mean = mean(ssimval_all, 1);

% Plot the statistics graphs
figure;
plot(threshold_vals, err_mean, 'LineWidth',2)
title('Mean Square Error Against Threshold Applied','fontsize',14)
xlabel('Threshold','fontsize',12)
ylabel('Mean Square Error','fontsize',12)

figure;
plot(threshold_vals, peaksnr_mean, 'LineWidth',2)
title('Peak-Signal-To-Noise Ratio Against Threshold Applied','fontsize',14)
xlabel('Threshold','fontsize',12)
ylabel('PSNR (dB)','fontsize',12)

figure;
plot(threshold_vals, ssimval_mean, 'LineWidth',2)
title('Structural Similarity Index Measure Against Threshold Applied','fontsize',14)
xlabel('Threshold','fontsize',12)
ylabel('SSIM','fontsize',12)