function refinedProposals = applyRegressionToProposals(obj, proposals, regressions)
    % APPLYREGRESSIONTOPROPOSALS Apply regression deltas to region proposals
    %
    % INPUTS:
    %   obj          - Object with InputSize property [height, width] or [height, width, channels]
    %   proposals    - dlarray of size [5, NumProposals, 1, BatchSize]
    %                  Format: [x1, y1, x2, y2, score; ...]
    %   regressions  - dlarray with supported formats:
    %                  [1, 1, 4, NumProposals] - single batch, single class
    %                  [1, 1, 4, NumProposals, BatchSize] - batched, single class
    %                  [4, NumProposals, 1, BatchSize] - standard format
    %                  [4*NumClasses, NumProposals, 1, BatchSize] - multi-class
    %                  Format: [dx, dy, dw, dh] deltas
    %
    % OUTPUT:
    %   refinedProposals - dlarray of same size as proposals with refined coordinates

    % Get image dimensions from object
    inputSize = obj.InputSize;
    if length(inputSize) >= 2
        imgHeight = inputSize(1);
        imgWidth = inputSize(2);
    else
        error('obj.InputSize must have at least 2 dimensions [height, width]');
    end
    
    % Default values
    classIdx = 1;
    minSize = 1;
    
    % Get dimensions
    [~, numProposals, ~, batchSize] = size(proposals);
    
    % Extract coordinates and scores
    x1 = proposals(1, :, :, :);
    y1 = proposals(2, :, :, :);
    x2 = proposals(3, :, :, :);
    y2 = proposals(4, :, :, :);
    scores = proposals(5, :, :, :);
    
    % Calculate current box properties
    widths = x2 - x1 + 1;
    heights = y2 - y1 + 1;
    centerX = x1 + 0.5 * (widths - 1);
    centerY = y1 + 0.5 * (heights - 1);
    
    % Handle different regression shapes and batch dimensions
    regSize = size(regressions);
    
    % Determine if we have batch dimension in regressions
    if length(regSize) == 4 && regSize(1) == 1 && regSize(2) == 1 && regSize(3) == 4
        % Format: [1, 1, 4, NumProposals] - single batch
        if batchSize > 1
            error('Regression format [1,1,4,N] requires BatchSize=1. Use format [1,1,4,N,BatchSize] for batches.');
        end
        if regSize(4) ~= numProposals
            error('Number of proposals in regression (%d) must match proposals (%d)', regSize(4), numProposals);
        end
        % Reshape to [4, NumProposals, 1, 1]
        regressions = reshape(regressions, [4, numProposals, 1, 1]);
        
    elseif length(regSize) == 5 && regSize(1) == 1 && regSize(2) == 1 && regSize(3) == 4
        % Format: [1, 1, 4, NumProposals, BatchSize] - batched
        if regSize(5) ~= batchSize
            error('Regression batch size (%d) must match proposal batch size (%d)', regSize(5), batchSize);
        end
        if regSize(4) ~= numProposals
            error('Number of proposals in regression (%d) must match proposals (%d)', regSize(4), numProposals);
        end
        % Reshape to [4, NumProposals, 1, BatchSize]
        regressions = reshape(regressions, [4, numProposals, 1, batchSize]);
        
    elseif length(regSize) == 4 && regSize(3) == 1 && regSize(4) == batchSize
        % Standard format: [4, NumProposals, 1, BatchSize] or [4*NumClasses, NumProposals, 1, BatchSize]
        % Already in correct format
        
    else
        error('Unsupported regression shape: [%s]. Expected formats: [1,1,4,N], [1,1,4,N,B], [4,N,1,B], or [4*C,N,1,B]', ...
              num2str(regSize));
    end
    
    % Extract regression deltas
    if size(regressions, 1) == 4
        % Single class regression
        dx = regressions(1, :, :, :);
        dy = regressions(2, :, :, :);
        dw = regressions(3, :, :, :);
        dh = regressions(4, :, :, :);
    else
        % Multi-class regression - extract specific class
        numClasses = size(regressions, 1) / 4;
        startIdx = (classIdx - 1) * 4 + 1;
        dx = regressions(startIdx, :, :, :);
        dy = regressions(startIdx + 1, :, :, :);
        dw = regressions(startIdx + 2, :, :, :);
        dh = regressions(startIdx + 3, :, :, :);
    end
    
    % Apply regression transformations
    % Standard R-CNN regression format:
    % dx = (pred_center_x - anchor_center_x) / anchor_width
    % dy = (pred_center_y - anchor_center_y) / anchor_height
    % dw = log(pred_width / anchor_width)
    % dh = log(pred_height / anchor_height)
    
    % Transform back to coordinates
    predCenterX = dx .* widths + centerX;
    predCenterY = dy .* heights + centerY;
    predWidth = exp(dw) .* widths;
    predHeight = exp(dh) .* heights;
    
    % Convert back to corner coordinates
    newX1 = predCenterX - 0.5 * (predWidth - 1);
    newY1 = predCenterY - 0.5 * (predHeight - 1);
    newX2 = predCenterX + 0.5 * (predWidth - 1);
    newY2 = predCenterY + 0.5 * (predHeight - 1);
    
    % Apply minimum size constraint and clipping
    % Ensure minimum width and height
    currentWidth = newX2 - newX1 + 1;
    currentHeight = newY2 - newY1 + 1;
    
    % Expand boxes that are too small
    tooSmallW = currentWidth < minSize;
    tooSmallH = currentHeight < minSize;
    
    if any(tooSmallW, 'all')
        expand = (minSize - currentWidth) / 2;
        newX1(tooSmallW) = newX1(tooSmallW) - expand(tooSmallW);
        newX2(tooSmallW) = newX2(tooSmallW) + expand(tooSmallW);
    end
    
    if any(tooSmallH, 'all')
        expand = (minSize - currentHeight) / 2;
        newY1(tooSmallH) = newY1(tooSmallH) - expand(tooSmallH);
        newY2(tooSmallH) = newY2(tooSmallH) + expand(tooSmallH);
    end
    
    % Clip to image boundaries using obj.InputSize
    newX1 = max(newX1, 0);
    newY1 = max(newY1, 0);
    newX2 = min(newX2, imgWidth - 1);
    newY2 = min(newY2, imgHeight - 1);
    
    % Round to whole numbers while preserving dlarray properties
    newX1 = round(newX1);
    newY1 = round(newY1);
    newX2 = round(newX2);
    newY2 = round(newY2);
    
    % Construct refined proposals
    refinedProposals = dlarray(zeros(5, numProposals, 1, batchSize, 'like', proposals));
    refinedProposals(1, :, :, :) = newX1;
    refinedProposals(2, :, :, :) = newY1;
    refinedProposals(3, :, :, :) = newX2;
    refinedProposals(4, :, :, :) = newY2;
    refinedProposals(5, :, :, :) = scores; % Keep original scores
    
end