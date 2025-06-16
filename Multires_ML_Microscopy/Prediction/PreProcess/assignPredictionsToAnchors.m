function featureMap = assignPredictionsToAnchors(predictedBoxes, anchorSizes, featureMapSize, varargin)
    arguments
        predictedBoxes (:,4) double % Px4 matrix with format [x1, y1, width, height]
        anchorSizes (:,2) double % Ax2 matrix with format [width, height]
        featureMapSize (1,3) double {mustBePositive, mustBeInteger} % [height, width, numAnchors]
        
        % Optional parameters
        varargin.StrideRatio (1,2) double = [1/16, 1/16] % Default stride ratio of 1/16
    end
    
    % Extract parameters from varargin
    strideRatio = varargin.StrideRatio;
    strideY = strideRatio(1);
    strideX = strideRatio(2);
    
    % Validate inputs further
    validateattributes(predictedBoxes, {'double'}, {'finite', 'nonnan', '>=', 0}, ...
        'assignPredictionsToAnchors', 'predictedBoxes');
    validateattributes(anchorSizes, {'double'}, {'finite', 'nonnan', '>', 0}, ...
        'assignPredictionsToAnchors', 'anchorSizes');
    
    % Check that number of anchor sizes matches the third dimension of featureMap
    if size(anchorSizes, 1) ~= featureMapSize(3)
        error('Number of anchor sizes (%d) must match the third dimension of featureMapSize (%d)', ...
            size(anchorSizes, 1), featureMapSize(3));
    end
    
    % Get dimensions
    height = featureMapSize(1);
    width = featureMapSize(2);
    numAnchors = featureMapSize(3);
    
    % Initialize feature map with zeros
    featureMap = zeros(featureMapSize);
    
    % For each predicted box
    for i = 1:size(predictedBoxes, 1)
        predBox = predictedBoxes(i, :); % [x1, y1, width, height] in image coords
        
        % Validate predicted box dimensions
        if predBox(3) <= 0 || predBox(4) <= 0
            warning('Skipping prediction %d due to non-positive width or height', i);
            continue;
        end
        
        % Calculate center of the box for feature map position mapping
        centerX = predBox(1) + predBox(3)/2;
        centerY = predBox(2) + predBox(4)/2;
        
        % Map from image coordinates to feature map coordinates
        featMapX = round(centerX * strideX);
        featMapY = round(centerY * strideY);
        
        % Keep within bounds
        featMapX = max(1, min(featMapX, width));
        featMapY = max(1, min(featMapY, height));
        
        % Find the best matching anchor for this prediction at this position
        bestIoU = -1;
        bestAnchorIdx = -1;
        
        for j = 1:numAnchors
            % Create an anchor box at this position with this anchor size
            anchorWidth = anchorSizes(j, 1);
            anchorHeight = anchorSizes(j, 2);
            
            % Create anchor box in format [x1, y1, width, height]
            anchorX1 = centerX - anchorWidth/2;
            anchorY1 = centerY - anchorHeight/2;
            anchorBox = [anchorX1, anchorY1, anchorWidth, anchorHeight];
            
            % Calculate IoU between prediction and this anchor
            iou = calculateIoU_TopLeft(predBox, anchorBox);
            
            if iou > bestIoU
                bestIoU = iou;
                bestAnchorIdx = j;
            end
        end
        
        % Set the corresponding location in feature map to 1
        featureMap(featMapY, featMapX, bestAnchorIdx) = 1;
    end
end

function iou = calculateIoU_TopLeft(boxA, boxB)
    % Calculate IoU between two bounding boxes
    % boxA, boxB format: [x1, y1, width, height] where (x1,y1) is top-left
    
    % Calculate coordinates for bottom-right corners
    boxA_x2 = boxA(1) + boxA(3);
    boxA_y2 = boxA(2) + boxA(4);
    
    boxB_x2 = boxB(1) + boxB(3);
    boxB_y2 = boxB(2) + boxB(4);
    
    % Calculate intersection
    xOverlap = max(0, min(boxA_x2, boxB_x2) - max(boxA(1), boxB(1)));
    yOverlap = max(0, min(boxA_y2, boxB_y2) - max(boxA(2), boxB(2)));
    intersection = xOverlap * yOverlap;
    
    % Calculate union
    areaA = boxA(3) * boxA(4);
    areaB = boxB(3) * boxB(4);
    union = areaA + areaB - intersection;
    
    % Calculate IoU
    iou = intersection / union;
end