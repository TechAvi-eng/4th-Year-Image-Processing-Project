clear all
close all
clc





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

db_wavelets = {'db1', 'db3', 'db5', 'db7', ...
               'db9', 'db11', 'db13', 'db15', ...
               'db17', 'db19',};

sym_wavelets = {'sym2', 'sym4', 'sym6', 'sym8', ...
                'sym10', 'sym12', 'sym14', ...
                'sym16', 'sym18', 'sym20'};

coif_wavelets = {'coif1', 'coif3', 'coif5'};

bior_wavelets = {'bior1.1', 'bior1.5', ...
                 'bior2.2', 'bior2.6', ...
                 'bior3.1', 'bior3.5', ...
                 'bior4.4', 'bior5.5', 'bior6.8'};

rbio_wavelets = {'rbio1.1', 'rbio1.5', ...
                 'rbio2.2', 'rbio2.6', ...
                 'rbio3.1', 'rbio3.5', ...
                 'rbio4.4', 'rbio5.5', 'rbio6.8'};

wavelets = [db_wavelets, sym_wavelets, coif_wavelets, bior_wavelets, rbio_wavelets];


% Define folder containing test images
folder_path = 'test_images';
imageFiles = dir(fullfile(folder_path, '*.png'));


NumberImages = 1;
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
        fprintf('\nUsing wavelet: %s\n', wavelet_name);
            
        tic;

        % Perform wavelet decomposition up to n levels with the current wavelet
        [C, S] = wavedec2(imgs, n, wavelet_name);

        % Apply wavelet function here
        
        elapsed_times(imageNumber, w) = toc;
        fprintf('The wavelet transform with %s took %.4f seconds.\n', wavelet_name, elapsed_times(w));


        for level = 1:n
            % Get detail coefficients for the current level
            [H, V, D] = detcoef2('all', C, S, level);
    
            % Get approximation coefficients for the current level
            A = appcoef2(C, S, wavelet_name, level);
            
            C_approx = C;
            
            % Zero out the detail coefficients for each level up to the specified level
            for j = 1:level
                % Determine the starting indices for the details at this level
                % Size of the current level's detail coefficients
                level_size = S(end-j+1, :);
                num_coeffs = prod(level_size);
    
                % Calculate starting indices for H, V, and D in the coefficient array C
                startIdxH = sum(prod(S(1:end-j, :), 2)) + 1;
                startIdxV = startIdxH + num_coeffs;
                startIdxD = startIdxV + num_coeffs;
    
                % Zero out the coefficients in the C array for horizontal, vertical, and diagonal details
                C_approx(startIdxH:startIdxH + num_coeffs - 1) = 0; % Horizontal detail
                C_approx(startIdxV:startIdxV + num_coeffs - 1) = 0; % Vertical detail
                C_approx(startIdxD:startIdxD + num_coeffs - 1) = 0; % Diagonal detail
            end
            
            % Reconstruct the image using only the approximation coefficients
            reconstructed_img = waverec2(C_approx, S, wavelet_name);

            % Resize the reconstructed image to match the original dimensions, if necessary
            if size(reconstructed_img) ~= size(imgs)
                reconstructed_img = imresize(reconstructed_img, size(imgs));
            end

            % Calculate SSIM between the original and reconstructed image
            ssim_val = ssim(double(reconstructed_img), imgs);
    
            % Calculate PSNR between the original and reconstructed image
            peaksnr_val = psnr(double(reconstructed_img)/ 255, double(imgs)/ 255);

            % Display the metric values
            disp(['Level ' num2str(level) ': SSIM = ' num2str(ssim_val)]);
            disp(['Level ' num2str(level) ': PSNR = ' num2str(peaksnr_val)]);

            ssim_values(imageNumber, w, level) = ssim_val; % Store the SSIM value
            psnr_values(imageNumber, w, level) = peaksnr_val; % Store the PSNR value

            % Display the reconstructed image
            % figure
            % imagesc(uint8(reconstructed_img))
            % colormap(gray)
    
            % title(['Reconstructed Image using Approximation Coefficients of Level ' num2str(level)])
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
