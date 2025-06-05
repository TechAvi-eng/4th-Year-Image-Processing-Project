clear all;
close all;
clc;

% Define folder containing test images
folder_path = 'test_images';
imageFiles = dir(fullfile(folder_path, '*.png'));

% Set up wavelet parameters
waveletType = 'haar'; % Use Daubechies 4 wavelet
numLevels = 3; % Number of decomposition levels
threshold_max = 10; % Maximum threshold value to iterate over

% Set the optimal threshold for visualization
optimal_threshold = 3;

% Initialize arrays to store metrics
numImages = length(imageFiles);
threshold_vals = 1:threshold_max;
err = zeros(numImages, threshold_max);
peaksnr = zeros(numImages, threshold_max);
ssimval = zeros(numImages, threshold_max);

for imageNumber = 1:5
    % Get the full image path
    imagePath = fullfile(folder_path, imageFiles(imageNumber).name);

    % Import image and convert to grayscale if necessary
    im = imread(imagePath);
    if size(im, 3) == 3
        imgs = mean(im, 3);
    else
        imgs = im;
    end
    
    imgs = double(imgs); % Convert to double for processing
    imgs = rescale(imgs); % Rescale original image for consistency
    
    % Perform DWT decomposition
    [c, s] = wavedec2(imgs, numLevels, waveletType);
    
    for threshold = 1:threshold_max
        % Soft thresholding on wavelet coefficients
        c_denoised = wthresh(c, 's', threshold);
        
        % Reconstruct the denoised image
        img_denoised = waverec2(c_denoised, s, waveletType);
        img_denoised = rescale(real(img_denoised)); % Rescale denoised image for consistency
        
        % Calculate error metrics
        err(imageNumber, threshold) = immse(img_denoised, imgs);
        peaksnr(imageNumber, threshold) = psnr(img_denoised, imgs);
        ssimval(imageNumber, threshold) = ssim(img_denoised, imgs);
        
        % Store the denoised image at the optimal threshold for display
        if threshold == optimal_threshold
            img_denoised_optimal = img_denoised;
        end
    end

    fprintf('Processed Image %d/%d\n', imageNumber, numImages);
end

% Calculate mean values across all images
err_mean = mean(err, 1);
peaksnr_mean = mean(peaksnr, 1);
ssimval_mean = mean(ssimval, 1);

% Plot the MSE values against the threshold values
figure;
plot(threshold_vals, err_mean, 'LineWidth', 2);
title('Mean Square Error (MSE) Against Threshold');
xlabel('Threshold');
ylabel('Mean Square Error');
grid on;

% Plot the PSNR values against the threshold values
figure;
plot(threshold_vals, peaksnr_mean, 'LineWidth', 2);
title('Peak Signal-to-Noise Ratio (PSNR) Against Threshold');
xlabel('Threshold');
ylabel('PSNR (dB)');
grid on;

% Plot the SSIM values against the threshold values
figure;
plot(threshold_vals, ssimval_mean, 'LineWidth', 2);
title('Structural Similarity Index Measure (SSIM) Against Threshold');
xlabel('Threshold');
ylabel('SSIM');
grid on;


% Display the original and denoised images for the final threshold
figure;
tiledlayout(1,2);

% Original Image
nexttile;
imshow(imgs, []);
title('Original Image');

% Denoised Image with optimal threshold
nexttile;
imshow(img_denoised_optimal, []);
title(['Denoised Image']);
