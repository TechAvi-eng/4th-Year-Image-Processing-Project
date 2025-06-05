function im = DWT_Denoise(im, Options)
% DWT_DENOISE De-noises image via DWT-thresholding on multiple levels of decomposition
%
% This function performs image denoising using Discrete Wavelet Transform (DWT)
% by applying soft thresholding to the detail coefficients at multiple decomposition levels.
% The denoising process involves:
% 1. Multi-level DWT decomposition
% 2. Soft thresholding of detail coefficients (horizontal, vertical, diagonal)
% 3. Reconstruction via inverse DWT

arguments
    im                                                                      % Input image to be denoised
    Options.Wavelet char = 'db5'                                           % Wavelet type (default: Daubechies 5)
    Options.Level (1,1) {mustBeInteger, mustBeReal} = 4                    % Number of decomposition levels
    Options.Threshold {mustBeGreaterThanOrEqual(Options.Threshold, 0), ...
                      mustBeLessThan(Options.Threshold, 1), ...
                      mustBeReal(Options.Threshold)} = 0.02                % Threshold factor(s) for denoising
end

% Validate and prepare threshold array for each decomposition level
Options.Threshold = iValidateThresholdLength(Options.Threshold, Options.Level);

% Convert scalar threshold to array for consistent processing across levels
if isscalar(Options.Threshold)
    thresholds = ones(1, Options.Level) * Options.Threshold;
else
    thresholds = Options.Threshold;
end

%% Forward DWT Decomposition and Thresholding
% Perform multi-level DWT decomposition, applying soft thresholding to detail coefficients
% at each level to remove noise while preserving important image features

coeffs = cell(Options.Level, 4);  % Storage for coefficients: [cA, cH, cV, cD]

for level = 1:Options.Level
    % Perform 2D DWT on image (level 1) or approximation coefficients (higher levels)
    if level == 1
        [cA, cH, cV, cD] = dwt2(im, Options.Wavelet);
    else
        [cA, cH, cV, cD] = dwt2(cA, Options.Wavelet);
    end
    
    % Calculate adaptive thresholds based on maximum coefficient values
    % This ensures threshold scales appropriately with signal strength
    current_threshold = thresholds(level);
    cH_threshold = current_threshold * max(abs(cH(:)));
    cV_threshold = current_threshold * max(abs(cV(:)));
    cD_threshold = current_threshold * max(abs(cD(:)));
    
    % Store coefficients: approximation (unchanged) and thresholded details
    coeffs{level, 1} = cA;                                    % Approximation coefficients (preserved)
    coeffs{level, 2} = wthresh(cH, 's', cH_threshold);       % Horizontal detail (soft thresholded)
    coeffs{level, 3} = wthresh(cV, 's', cV_threshold);       % Vertical detail (soft thresholded)
    coeffs{level, 4} = wthresh(cD, 's', cD_threshold);       % Diagonal detail (soft thresholded)
end

%% Inverse DWT Reconstruction
% Reconstruct the denoised image by performing inverse DWT from highest
% to lowest decomposition level using the thresholded coefficients

% Start reconstruction from the deepest approximation coefficients
cA_reconstructed = coeffs{Options.Level, 1};

% Reconstruct from highest level down to level 2
for level = Options.Level:-1:2
    % Retrieve thresholded detail coefficients for current level
    cH = coeffs{level, 2};
    cV = coeffs{level, 3};
    cD = coeffs{level, 4};
    
    % Ensure dimensional consistency between approximation and detail coefficients
    % This handles potential size mismatches due to image dimensions
    [cA_reconstructed] = adjustCoefficientSizes(cA_reconstructed, cD);
    
    % Perform inverse DWT to get approximation for next level up
    cA_reconstructed = idwt2(cA_reconstructed, cH, cV, cD, Options.Wavelet);
    
    % Store reconstructed approximation for potential intermediate access
    coeffs{level-1, 1} = cA_reconstructed;
end

% Final reconstruction: convert level 1 coefficients back to image domain
cA = coeffs{1, 1};
cH = coeffs{1, 2};
cV = coeffs{1, 3};
cD = coeffs{1, 4};

% Final size adjustment before image reconstruction
[cA] = adjustCoefficientSizes(cA, cD);

% Reconstruct the final denoised image
im = idwt2(cA, cH, cV, cD, Options.Wavelet);

end

function coeff_adjusted = adjustCoefficientSizes(coeff_a, coeff_d)
% ADJUSTCOEFFICIENTSIZES Ensures coefficient matrices have compatible dimensions
% Removes excess rows/columns from approximation coefficients to match detail coefficients
    
    coeff_adjusted = coeff_a;
    
    [a_rows, a_cols] = size(coeff_a);
    [d_rows, d_cols] = size(coeff_d);
    
    % Adjust columns if sizes don't match
    if a_cols ~= d_cols
        coeff_adjusted(:, a_cols) = [];
    end
    
    % Adjust rows if sizes don't match  
    if a_rows ~= d_rows
        coeff_adjusted(a_rows, :) = [];
    end
end

function thresholds = iValidateThresholdLength(thresholds, levels)
% IVALIDATETHRESHOLDLENGTH Validates and formats threshold parameter
% Ensures threshold input is either scalar or vector matching decomposition levels
    
    [r, c] = size(thresholds);
    
    % Handle case where user provides column vector instead of row vector
    if ~(r==1) && (c==1)
        thresholds = thresholds';
        [r, c] = size(thresholds);  % Update dimensions after transpose
    end
    
    % Validate that threshold dimensions are acceptable
    valid_scalar = (r==1) && (c==1);                    % Single threshold for all levels
    valid_row_vector = (r==1) && (c==levels);           % One threshold per level (row)
    valid_col_vector = (r==levels) && (c==1);           % One threshold per level (column)
    
    if ~(valid_scalar || valid_row_vector || valid_col_vector)
        error('Thresholds must be of size (1,1) or (1,levels) or (levels,1)');
    end
end