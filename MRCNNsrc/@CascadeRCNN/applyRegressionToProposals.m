function refinedProposals = applyRegressionToProposals(proposals, regressions)
    % APPLYREGRESSIONTOPROPOSALS Apply regression deltas to region proposals
    %
    % INPUTS:
    %   proposals    - dlarray of size [5, NumProposals, 1, BatchSize]
    %                  Format: [x1, y1, x2, y2, score; ...]
    %   regressions  - dlarray with supported formats:
    %                  [1, 1, 1, NumProposals*4] - single batch, single class
    %                  [1, 1, 1, NumProposals*4, BatchSize] - batched, single class
    %                  [4, NumProposals, 1, BatchSize] - standard format
    %                  [4*NumClasses, NumProposals, 1, BatchSize] - multi-class
    %                  Format: [dx, dy, dw, dh] deltas
    %
    % OPTIONAL PARAMETERS:
    %   'ClassIndex' - Which class regression to use (default: 1)
    %   'ClipBounds' - [width, height] to clip boxes (default: no clipping)
    %   'MinSize'    - Minimum box size (default: 1)
    %
    % OUTPUT:
    %   refinedProposals - dlarray of same size as proposals with refined coordinates

    % Parse input arguments
    p = inputParser;
    addRequired(p, 'proposals', @(x) isa(x, 'dlarray'));
    addRequired(p, 'regressions', @(x) isa(x, 'dlarray'));
    addParameter(p, 'ClassIndex', 1, @(x) isscalar(x) && x > 0);
    addParameter(p, 'ClipBounds', [], @(x) isempty(x) || (isnumeric(x) && length(x) == 2));
    addParameter(p, 'MinSize', 1, @(x) isscalar(x) && x > 0);
    
    classIdx = p.Results.ClassIndex;
    clipBounds = p.Results.ClipBounds;
    minSize = p.Results.MinSize;
    
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
    if length(regSize) == 4 && regSize(1) == 1 && regSize(2) == 1 && regSize(3) == 1
        % Format: [1, 1, 1, NumProposals*4] - single batch
        if batchSize > 1
            error('Regression format [1,1,1,N] requires BatchSize=1. Use format [1,1,1,N,BatchSize] for batches.');
        end
        totalRegs = regSize(4);
        if totalRegs == numProposals * 4
            % Single class: reshape to [4, NumProposals, 1, 1]
            regressions = reshape(regressions, [4, numProposals, 1, 1]);
        else
            % Multi-class: reshape accordingly
            numClasses = totalRegs / (numProposals * 4);
            regressions = reshape(regressions, [4*numClasses, numProposals, 1, 1]);
        end
        
    elseif length(regSize) == 5 && regSize(1) == 1 && regSize(2) == 1 && regSize(3) == 1
        % Format: [1, 1, 1, NumProposals*4, BatchSize] - batched
        if regSize(5) ~= batchSize
            error('Regression batch size (%d) must match proposal batch size (%d)', regSize(5), batchSize);
        end
        totalRegs = regSize(4);
        if totalRegs == numProposals * 4
            % Single class: reshape to [4, NumProposals, 1, BatchSize]
            regressions = reshape(regressions, [4, numProposals, 1, batchSize]);
        else
            % Multi-class: reshape accordingly
            numClasses = totalRegs / (numProposals * 4);
            regressions = reshape(regressions, [4*numClasses, numProposals, 1, batchSize]);
        end
        
    elseif length(regSize) == 4 && regSize(3) == 1 && regSize(4) == batchSize
        % Standard format: [4, NumProposals, 1, BatchSize] or [4*NumClasses, NumProposals, 1, BatchSize]
        % Already in correct format
        
    else
        error('Unsupported regression shape: [%s]. Expected formats: [1,1,1,N], [1,1,1,N,B], [4,N,1,B], or [4*C,N,1,B]', ...
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
    // dy = (pred_center_y - anchor_center_y) / anchor_height
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
    
    % Apply minimum size constraint
    if minSize > 1
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
    end
    
    % Clip to image boundaries if specified
    if ~isempty(clipBounds)
        imgWidth = clipBounds(1);
        imgHeight = clipBounds(2);
        
        newX1 = max(newX1, 0);
        newY1 = max(newY1, 0);
        newX2 = min(newX2, imgWidth - 1);
        newY2 = min(newY2, imgHeight - 1);
    end
    
    % Construct refined proposals
    refinedProposals = dlarray(zeros(5, numProposals, 1, batchSize, 'like', proposals));
    refinedProposals(1, :, :, :) = newX1;
    refinedProposals(2, :, :, :) = newY1;
    refinedProposals(3, :, :, :) = newX2;
    refinedProposals(4, :, :, :) = newY2;
    refinedProposals(5, :, :, :) = scores; % Keep original scores
    
end