function thresh_reconstructed_img = DWT_Denoise(img, Options)
    % Default argument values specified
    arguments
        img
        Options.Wavelet = 'db5' % default value
        Options.Level = 4 % default value
        Options.Threshold = 0.1; % default value
    end

    % If input is a scalar value then create an array
    if isscalar(Options.Threshold)
        threshold = ones(1, Options.Level) * Options.Threshold;
    else
        threshold = Options.Threshold;

function thresh_reconstructed_img = Coefficients(img, wavelet, n, threshold_in)
    % OPTIONAL arguments:
    % Positional arguments means arguments position must be filled in
    % function call to identify arguments that come after it

    % Default argument values specified
    arguments
        img
        wavelet = 'db5' % default value
        n = 4 % default value
        threshold_in = 0.1; % default value
    end

    % If input is a scalar value then create an array
    if isscalar(threshold_in)
        threshold = ones(1, n) * threshold_in;
    else
        threshold = threshold_in;

    end
    
    
    % If input is a scalar value then create an array
    if isscalar(Options.Threshold)
        threshold = ones(1, Options.Level) * Options.Threshold;
    else
        % If input is already an array
        threshold = Options.Threshold;
    end

    if isscalar(threshold_in)
        threshold = ones(1, n) * threshold_in;
    else
        % If input is already an array
        threshold = threshold_in;

    end

    %% Discrete Wavelet Transform    
    % Loops through each decomposition level calculating coefficients from
    % previous level approximation coefficients
    for level = 1:Options.Level
        if level == 1
            [cA, cH, cV, cD] = dwt2(img, Options.Wavelet);
        else
            [cA, cH, cV, cD] = dwt2(cA, Options.Wavelet);
    
    for level = 1:n
        if level == 1
            [cA, cH, cV, cD] = dwt2(img, wavelet);
        else
            [cA, cH, cV, cD] = dwt2(cA, wavelet);

    end
        
        % Dynamic threshold which can be different for each decomposition level 
        t = threshold(level);

        % Calculating thresholding parameters
        cH_max = t * max(abs([cH(:)]));
        cV_max = t * max(abs([cV(:)]));
        cD_max = t * max(abs([cD(:)]));

        coeffs{level, 1} = cA;
        coeffs{level, 2} = wthresh(cH,'s',cH_max);
        coeffs{level, 3} = wthresh(cV,'s',cV_max);
        coeffs{level, 4} = wthresh(cD,'s',cD_max);

    end
    
    %% Reconstruction
    % Obtaining highest decomposition level approximation coefficients
    cA_rec = coeffs{Options.Level, 1};

    cA_rec = coeffs{n, 1};

    % Loops through each decomposition level (highest level of 
    % decomposition to lowest) and calculates the reconstructed
    % approximation coefficients for the next level
    for level = Options.Level:-1:2
        cH = coeffs{level, 2};
        cV = coeffs{level, 3};
        cD = coeffs{level, 4};
        cA_rec = idwt2(cA_rec, cH, cV, cD, Options.Wavelet);

    for level = n:-1:2
        cH = coeffs{level, 2};
        cV = coeffs{level, 3};
        cD = coeffs{level, 4};
        
        % Ensuring cA and cD have the same number of columns
        cA_cols = size(cA_rec,2);
        cD_cols = size(cH,2);

        if cA_cols ~= cD_cols
            cA_rec(:,cA_cols) = [];
        end
        
        % Inverse DWT to reconstruct cA
        cA_rec = idwt2(cA_rec, cH, cV, cD, wavelet);

        coeffs{level-1, 1} = cA_rec;
    end

    % Reconstruct the image using the thresholded coefficients (level 1
    % decomposition coefficients)
    cA = coeffs{1, 1};
    cH = coeffs{1, 2};
    cV = coeffs{1, 3};
    cD = coeffs{1, 4};

    thresh_reconstructed_img = idwt2(cA, cH, cV, cD, Options.Wavelet);

    thresh_reconstructed_img = idwt2(cA, cH, cV, cD, wavelet);

    
    % Display the original and reconstructed images
    figure;
    subplot(1,2,1);
    imshow(img, []);
    title('Original Image');

    subplot(1,2,2);
    imshow(thresh_reconstructed_img, []);
    title('Thresholded Reconstructed Image');
    
end
