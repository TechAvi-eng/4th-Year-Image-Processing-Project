function im = DWT_Denoise(im, Options)
% DWT_DENOISE De-noises image via DWT-thresholding on multiple levels of decomposition
    arguments
        im
        Options.Wavelet char = 'db5' % default value
        Options.Level (1,1) {mustBeInteger, mustBeReal} = 4 % default value
        Options.Threshold {mustBeGreaterThanOrEqual(Options.Threshold, 0), mustBeLessThan(Options.Threshold, 1), mustBeReal(Options.Threshold)}= 0.02; % default value
    end

    Options.Threshold = iValidateThresholdLength(Options.Threshold, Options.Level);

    % If input is a scalar value then create an array
    if isscalar(Options.Threshold)
        threshold = ones(1, Options.Level) * Options.Threshold;
    else
        threshold = Options.Threshold;
    end
    
    
    % If input is a scalar value then create an array
    if isscalar(Options.Threshold)
        threshold = ones(1, Options.Level) * Options.Threshold;
    else
        % If input is already an array
        threshold = Options.Threshold;
    end

    %% Discrete Wavelet Transform    
    % Loops through each decomposition level calculating coefficients from
    % previous level approximation coefficients
    for level = 1:Options.Level
        if level == 1
            [cA, cH, cV, cD] = dwt2(im, Options.Wavelet);
        else
            [cA, cH, cV, cD] = dwt2(cA, Options.Wavelet);
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

    % Loops through each decomposition level (highest level of 
    % decomposition to lowest) and calculates the reconstructed
    % approximation coefficients for the next level
    for level = Options.Level:-1:2
        cH = coeffs{level, 2};
        cV = coeffs{level, 3};
        cD = coeffs{level, 4};
        
        % Ensuring cA and cD have the same number of columns and rows
        cA_cols = size(cA_rec,2);
        cA_rows = size(cA_rec,1);
        cD_cols = size(cD,2);
        cD_rows = size(cD,1);

        if cA_cols ~= cD_cols
            cA_rec(:,cA_cols) = [];
        end

        if cA_rows ~= cD_rows
            cA_rec(cA_rows,:) = [];
        end
        
        % Inverse DWT to reconstruct cA
        cA_rec = idwt2(cA_rec, cH, cV, cD, Options.Wavelet);

        coeffs{level-1, 1} = cA_rec;
    end
    


    % Reconstruct the image using the thresholded coefficients (level 1
    % decomposition coefficients) 
    % Additionally, checking the row and column size of Approximation
    % coefficients
    cA = coeffs{1, 1};
    cA_cols = size(cA,2);
    cA_rows = size(cA,1);
    cH = coeffs{1, 2};
    cV = coeffs{1, 3};
    cD = coeffs{1, 4};
    cD_cols = size(cD,2);
    cD_rows = size(cD,1);

    % Amending Approximation Coefficient row and column size if required
    if cA_cols ~= cD_cols
        cA(:,cA_cols) = [];
        
    end

    if cA_rows ~= cD_rows
        cA(cA_rows,:) = [];
    end

    % Performing inverse DWT
    im = idwt2(cA, cH, cV, cD, Options.Wavelet);

end
function thresholds = iValidateThresholdLength(thresholds, levels)
    [r c] = size(thresholds);
    if ~r==1 & c==1   %contingency if user enters transpose of desired shape
        thresholds = thresholds';
    end
    if ~(and(r==1,c==1) || and(r==1,c==levels) || and(r==levels,c==1))
        error('Thresholds must be of size (1,1) or (1,levels)');
    end
end
