clear all
close all
clc

% Define folder containing test images
folder_path = 'test_images';
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
    wavelet = 'db5';

    % Perform wavelet decomposition up to n levels
    % C = wavelet decomposition vector
    % S = bookkeeping matrix
    [C, S] = wavedec2(imgs, n, wavelet);
    
    for level = 1:n
        % Get detail coefficients for the current level
        [H, V, D] = detcoef2('all', C, S, level);

        % Get approximation coefficients for the current level
        A = appcoef2(C, S, wavelet, level);
        
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
        reconstructed_img = waverec2(C_approx, S, wavelet);
        
        % Resize the reconstructed image to match the original dimensions, if necessary
        if size(reconstructed_img) ~= size(imgs)
            reconstructed_img = imresize(reconstructed_img, size(imgs));
        end
        
        % Calculate SSIM and PSNR between the original and reconstructed image
        ssim_val = ssim(double(reconstructed_img), imgs);
        psnr_val = psnr(double(reconstructed_img)/ 255, double(imgs)/ 255);

        % Display the SSIM and PSNR value
        disp(['Level ' num2str(level) ': SSIM = ' num2str(ssim_val)]);
        disp(['Level ' num2str(level) ': PSNR = ' num2str(psnr_val)]);
    
        % Display the reconstructed image
        figure
        imagesc(uint8(reconstructed_img))
        colormap(gray)

        title(['Reconstructed Image using Approximation Coefficients of Level ' num2str(level)])

    end

end