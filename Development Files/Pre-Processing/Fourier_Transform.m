clear all
close all
clc

% Define folder containing test images
folder_path = 'test_images';
imageFiles = dir(fullfile(folder_path, '*.png'));

for imageNumber = 26:26 %length(imageFiles)
    % Get the full image path
    imagePath = fullfile(folder_path, imageFiles(imageNumber).name);

    %%
    % Importing image
    im=imread(imagePath);
    
    % % Displaying the original image
    % figure
    % imshow(im)
    % title('Original')
    

    %%
    % Coverts image to grayscale by taking mean of the 3 colour channels
    imgs=mean(im,3);
    
    
    %%
    % Apply Fourier Transform
    ft=fft2(imgs); % Takes the 2-D fft
    
    % Shift zero frequency to centre
    ft=fftshift(ft);
    
    % Compute magnitude of complex numbers
    magnitude_ft = abs(ft);
    
    % Log scale results
    log_magnitude_ft = log(1 + magnitude_ft);
    
    
    %%
    % Applying a high-pass (hp) filter
    
    % Get image size
    [rows, cols] = size(imgs);
    
    % Create a meshgrid (for distance calculations from centre)
    [u, v] = meshgrid(1:cols, 1:rows);
    
    % Calculate the center of the frequency domain
    center_u = ceil(cols / 2);
    center_v = ceil(rows / 2);
    
    % Compute distance from the center
    D = sqrt((u - center_u).^2 + (v - center_v).^2);
    
    % Defining the maximum frequency threshold for removal
    threshold_max = 100;

    % Defining index i
    i = 1;
    
    for threshold = 100:threshold_max  
        threshold_vals(i) = threshold;
        
        % threshold is the radius of the low frequency region to block
    
        % Create a binary mask for high-pass filtering (>)
        % Create a binary mask for low-pass filtering (<)
        % 1 if outside filter, 0 if inside filter
        high_pass_filter = double(D < threshold);
    
        % Apply high-pass filter in the frequency domain
        ft_filtered = ft .* high_pass_filter;
        
        % For displaying filtered Fourier Transform
        magnitude_ft_filtered = abs(ft_filtered);
        log_magnitude_ft_filtered = log(1 + magnitude_ft_filtered);
    
        % Perform inverse FFT to get filtered image
        ift = ifftshift(ft_filtered); % Shift the zero frequency back
        img_filtered = ifft2(ift); % Inverse Fourier Transform
        
        % Take the real part of the inverse FFT result
        img_filtered = real(img_filtered);
        
        % Rescaling images
        imgs = rescale(imgs);
        img_filtered = rescale(img_filtered);
    
        % Set default font and interpreter globally for consistency
        set(groot, 'DefaultTextInterpreter', 'latex');
        set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
        set(groot, 'DefaultLegendInterpreter', 'latex');
        set(groot, 'DefaultAxesFontName', 'CMU Serif');
        set(groot, 'DefaultTextFontName', 'CMU Serif');
        
        % Create tiled layout
        figure;
        t = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
        
        % Displaying original grayscale image
        nexttile(1);
        imshow(imgs);
        title('Grayscale Original Image');
        
        % Display Fourier transform
        nexttile(2);
        imagesc(log_magnitude_ft);
        colormap(gca, jet);
        colorbar;
        title('Fourier Transform of Image');
        xlabel('Frequency X (Hz)');
        ylabel('Frequency Y (Hz)');
        
        % Displaying filtered Fourier transform
        nexttile(4);
        imagesc(log_magnitude_ft_filtered);
        colormap(gca, jet);
        colorbar;
        title('Filtered Fourier Transform (Low-Pass)');
        xlabel('Frequency X (Hz)');
        ylabel('Frequency Y (Hz)');
        
        % Display the filtered image
        nexttile(3);
        imshow(img_filtered);
        title('Image After Low-Pass Filtering');
       
        drawnow
        pause(5/1000)
        
        err(imageNumber, i) = immse(img_filtered, imgs);
        peaksnr(imageNumber, i) = psnr(img_filtered, imgs);
        ssimval(imageNumber, i) = ssim(img_filtered, imgs);
    
        i = i + 1;
    end

end

% Calculate mean of statistics extracted
err_mean = mean(err, 1);
peaksnr_mean = mean(peaksnr, 1);
ssimval_mean = mean(ssimval, 1);

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
