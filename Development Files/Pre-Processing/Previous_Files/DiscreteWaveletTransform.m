clear all
close all
clc

% Define folder containing test images
folder_path = 'RPE_Dataset';
imageFiles = dir(fullfile(folder_path, '*.png'));

% Change to for imageNumber = 1:1 for only one image analysis
% Change to for imageNumber = 1:length(imageFiles) for only one image analysis

for imageNumber = 1:1
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
    imgs = mean(im,3);
    
    
    %%
    % Apply Discrete Wavlet Transform
    
    % Levels of decomposition
    n = 5;

    % Perform wavelet decomposition up to n levels
    % C = wavelet decomposition vector
    % S = bookkeeping matrix
    [C, S] = wavedec2(imgs, n, 'db8');
    
    for level = 1:n
        % Get detail coefficients for the current level
        [H, V, D] = detcoef2('all', C, S, level);

        % Get approximation coefficients for the current level
        A = appcoef2(C, S, 'db8', level);
        
        % Scale coefficients for display
        H_img = wcodemat(H, 255, 'mat', 1);
        V_img = wcodemat(V, 255, 'mat', 1);
        D_img = wcodemat(D, 255, 'mat', 1);
        A_img = wcodemat(A, 255, 'mat', 1);
        
        % Display the coefficients for the current level
        figure
        subplot(2, 2, 1)
        imagesc(A_img)
        colormap(gray)
        title(['Approximation Coef. of Level ' num2str(level)])
        
        subplot(2, 2, 2)
        imagesc(H_img)
        title(['Horizontal Detail Coef. of Level ' num2str(level)])
        
        subplot(2, 2, 3)
        imagesc(V_img)
        title(['Vertical Detail Coef. of Level ' num2str(level)])
        
        subplot(2, 2, 4)
        imagesc(D_img)
        title(['Diagonal Detail Coef. of Level ' num2str(level)])
        
        % Reconstruct and display the approximation image for this level
        reconstruction = idwt2(A_img, [], [], [], 'db8');
        figure
        imshow(uint8(reconstruction))
        title(['Reconstructed Image from Level ' num2str(level)])
    end

end

% % Calculate mean of statistics extracted
% err_mean = mean(err, 1);
% peaksnr_mean = mean(peaksnr, 1);
% ssimval_mean = mean(ssimval, 1);
% 
% % Plot the statistics graphs
% figure;
% plot(threshold_vals, err_mean, 'LineWidth',2)
% title('Mean Square Error Against Threshold Applied','fontsize',14)
% xlabel('Threshold','fontsize',12)
% ylabel('Mean Square Error','fontsize',12)
% 
% figure;
% plot(threshold_vals, peaksnr_mean, 'LineWidth',2)
% title('Peak-Signal-To-Noise Ratio Against Threshold Applied','fontsize',14)
% xlabel('Threshold','fontsize',12)
% ylabel('PSNR (dB)','fontsize',12)
% 
% figure;
% plot(threshold_vals, ssimval_mean, 'LineWidth',2)
% title('Structural Similarity Index Measure Against Threshold Applied','fontsize',14)
% xlabel('Threshold','fontsize',12)
% ylabel('SSIM','fontsize',12)
