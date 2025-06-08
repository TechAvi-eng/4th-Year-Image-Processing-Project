function [tracks, trackingTable] = createTracks(tracks, detections, scores, frameIdx, trackingTable, args)
    % This function processes a single frame's detections and updates the tracks
    % 
    % Inputs:
    %   tracks - Current tracks structure (empty for the first frame)
    %   detections - Nx4 array of bounding boxes [x,y,width,height] for current frame
    %   scores - Nx1 array of confidence scores (0-1) for each detection
    %   frameIdx - Current frame number
    %   trackingTable - Table to store tracking results (can be empty for first frame)
    %   args - Name-value arguments for tracking parameters
    %
    % Optional parameters:
    %   'MinIoU' - Minimum IoU threshold for matching tracks (default: 0.3)
    %   'MaxInvisibleCount' - Maximum number of consecutive frames a track can be missing (default: 10)
    %   'SizeWeight' - Weight for size difference in cost calculation (default: 0.2)
    %   'AspectRatioWeight' - Weight for aspect ratio difference in cost calculation (default: 0.2)
    %   'IoUWeight' - Weight for IoU component in cost calculation (default: 0.6)
    %   'MaxDistance' - Maximum distance for spatial gating (default: inf)
    %   'PreallocateRows' - Number of rows to preallocate in tracking table (default: 1000)
    %
    % Outputs:
    %   tracks - Updated tracks structure
    %   trackingTable - Updated tracking table with new results
    
    arguments
        tracks
        detections (:,4) double
        scores (:,1) double
        frameIdx (1,1) double
        trackingTable
        args.MinIoU (1,1) double = 0.3
        args.MaxInvisibleCount (1,1) double = 10
        args.SizeWeight (1,1) double = 0.2
        args.AspectRatioWeight (1,1) double = 0.2
        args.IoUWeight (1,1) double = 0.6
        args.MaxDistance (1,1) double = inf
        args.PreallocateRows (1,1) double = 1000
    end
    
    % Validate weights sum to 1
    totalWeight = args.SizeWeight + args.AspectRatioWeight + args.IoUWeight;
    if abs(totalWeight - 1) > 1e-6
        warning('Weights do not sum to 1. Normalizing weights.');
        args.SizeWeight = args.SizeWeight / totalWeight;
        args.AspectRatioWeight = args.AspectRatioWeight / totalWeight;
        args.IoUWeight = args.IoUWeight / totalWeight;
    end
    
    % Extract parameters from arguments
    iouThreshold = args.MinIoU;
    invisibleForTooLong = args.MaxInvisibleCount;
    maxDistance = args.MaxDistance;
    
    % Initialize tracks if this is the first frame
    if isempty(tracks)
        tracks = struct('id', {}, 'bbox', {}, 'score', {}, 'age', {}, ...
                      'totalVisibleCount', {}, 'consecutiveInvisibleCount', {});
        nextId = 1;
    else
        % Get the maximum ID from existing tracks (vectorized)
        if isempty([tracks.id])
            nextId = 1;
        else
            nextId = max([tracks.id]) + 1;
        end
    end
    
    % Initialize tracking table with preallocation
    if isempty(trackingTable) || (~istable(trackingTable))
        trackingTable = table('Size', [args.PreallocateRows 4], ...
                          'VariableTypes', {'double', 'double', 'double', 'cell'}, ...
                          'VariableNames', {'frameID', 'objectID', 'confidence', 'bbox'});
        trackingTable.Properties.UserData.actualRows = 0; % Keep track of actual rows used
    end
    
    % Handle empty detections case
    if isempty(detections)
        % Vectorized update of all existing tracks
        if ~isempty(tracks)
            ages = [tracks.age];
            invisibleCounts = [tracks.consecutiveInvisibleCount];
            
            % Update all tracks at once
            ageCell = num2cell(ages + 1);
            invisibleCell = num2cell(invisibleCounts + 1);
            [tracks.age] = ageCell{:};
            [tracks.consecutiveInvisibleCount] = invisibleCell{:};
            
            % Remove dead tracks (vectorized)
            isDead = [tracks.consecutiveInvisibleCount] >= invisibleForTooLong;
            tracks = tracks(~isDead);
        end
        return;
    end
    
    % Skip association if no current tracks
    if isempty(tracks)
        % Vectorized creation of new tracks
        numDets = size(detections, 1);
        ids = nextId:(nextId + numDets - 1);
        
        % Create all tracks at once - ensure proper structure array creation
        tracks = struct([]);  % Initialize empty struct array
        for i = 1:numDets
            tracks(i).id = ids(i);
            tracks(i).bbox = detections(i, :);
            tracks(i).score = scores(i);
            tracks(i).age = 1;
            tracks(i).totalVisibleCount = 1;
            tracks(i).consecutiveInvisibleCount = 0;
        end
        
        % Add to tracking table efficiently
        trackingTable = addToTrackingTable(trackingTable, frameIdx, ids, scores, detections);
        
        return;
    end
    
    % OPTIMIZATION: Spatial gating - only compute costs for nearby tracks
    numDetections = size(detections, 1);
    numTracks = length(tracks);
    
    % Extract track centers and detection centers for distance calculation
    trackBboxes = reshape([tracks.bbox], 4, [])'; % 4 x numTracks -> numTracks x 4
    trackCenters = trackBboxes(:, 1:2) + trackBboxes(:, 3:4) / 2; % [x + w/2, y + h/2]
    detCenters = detections(:, 1:2) + detections(:, 3:4) / 2;
    
    % Compute pairwise distances (vectorized)
    distMatrix = pdist2(trackCenters, detCenters);
    validPairs = distMatrix <= maxDistance;
    
    % Initialize cost matrix with high values
    costMatrix = inf(numTracks, numDetections);
    
    % Only compute costs for valid pairs
    [trackIndices, detIndices] = find(validPairs);
    
    if ~isempty(trackIndices)
        % Vectorized cost computation for valid pairs only
        trackBboxesValid = trackBboxes(trackIndices, :);
        detectionsValid = detections(detIndices, :);
        
        % Compute all components vectorized
        track_sizes = trackBboxesValid(:, 3) .* trackBboxesValid(:, 4);
        track_aspects = trackBboxesValid(:, 3) ./ trackBboxesValid(:, 4);
        det_sizes = detectionsValid(:, 3) .* detectionsValid(:, 4);
        det_aspects = detectionsValid(:, 3) ./ detectionsValid(:, 4);
        
        % Vectorized IoU computation
        ious = bboxIoUVectorized(trackBboxesValid, detectionsValid);
        iou_costs = 1 - ious;
        
        % Vectorized size and aspect ratio differences
        size_diffs = abs(track_sizes - det_sizes) ./ max(track_sizes, det_sizes);
        aspect_diffs = abs(track_aspects - det_aspects) ./ (1 + abs(track_aspects - det_aspects));
        
        % Combine costs
        costs = args.IoUWeight * iou_costs + ...
                args.SizeWeight * size_diffs + ...
                args.AspectRatioWeight * aspect_diffs;
        
        % Assign costs to matrix
        linearIndices = sub2ind(size(costMatrix), trackIndices, detIndices);
        costMatrix(linearIndices) = costs;
    end
    
    % Use assignment algorithm
    [assignments, unassignedTracks, unassignedDetections] = ...
        assignDetectionsToTracks(costMatrix, iouThreshold);
    
    % Update assigned tracks (vectorized where possible)
    if ~isempty(assignments)
        trackIndices = assignments(:, 1);
        detectionIndices = assignments(:, 2);
        
        % Update bounding boxes
        for i = 1:length(trackIndices)
            tracks(trackIndices(i)).bbox = detections(detectionIndices(i), :);
            tracks(trackIndices(i)).score = scores(detectionIndices(i));
            tracks(trackIndices(i)).age = tracks(trackIndices(i)).age + 1;
            tracks(trackIndices(i)).totalVisibleCount = tracks(trackIndices(i)).totalVisibleCount + 1;
            tracks(trackIndices(i)).consecutiveInvisibleCount = 0;
        end
        
        % Add to tracking table
        assignedIds = [tracks(trackIndices).id];
        assignedScores = scores(detectionIndices);
        assignedDetections = detections(detectionIndices, :);
        trackingTable = addToTrackingTable(trackingTable, frameIdx, assignedIds, assignedScores, assignedDetections);
    end
    
    % Update unassigned tracks (vectorized)
    if ~isempty(unassignedTracks)
        for i = 1:length(unassignedTracks)
            idx = unassignedTracks(i);
            tracks(idx).age = tracks(idx).age + 1;
            tracks(idx).consecutiveInvisibleCount = tracks(idx).consecutiveInvisibleCount + 1;
        end
    end
    
    % Create new tracks for unassigned detections - FIXED: Use vertical concatenation
    if ~isempty(unassignedDetections)
        numNewTracks = length(unassignedDetections);
        newIds = nextId:(nextId + numNewTracks - 1);
        
        newDetections = detections(unassignedDetections, :);
        newScores = scores(unassignedDetections);
        
        % Create new tracks structure array properly
        newTracks = struct([]);  % Initialize empty struct array
        for i = 1:numNewTracks
            newTracks(i).id = newIds(i);
            newTracks(i).bbox = newDetections(i, :);
            newTracks(i).score = newScores(i);
            newTracks(i).age = 1;
            newTracks(i).totalVisibleCount = 1;
            newTracks(i).consecutiveInvisibleCount = 0;
        end
        
        % FIXED: Use vertical concatenation to append new tracks
        if isempty(tracks)
            tracks = newTracks;
        else
            tracks = [tracks, newTracks];  % Vertical concatenation
        end
        
        % Add to tracking table
        trackingTable = addToTrackingTable(trackingTable, frameIdx, newIds, newScores, newDetections);
    end
    
    % Remove dead tracks (vectorized)
    if ~isempty(tracks)
        isDead = [tracks.consecutiveInvisibleCount] >= invisibleForTooLong;
        tracks = tracks(~isDead);
    end
end

% Optimized function to add entries to tracking table
function trackingTable = addToTrackingTable(trackingTable, frameIdx, ids, scores, detections)
    numEntries = length(ids);
    currentRows = trackingTable.Properties.UserData.actualRows;
    
    % Check if we need to expand the table
    if currentRows + numEntries > height(trackingTable)
        % Double the table size
        newSize = max(height(trackingTable) * 2, currentRows + numEntries);
        emptyRows = newSize - height(trackingTable);
        
        emptyTable = table('Size', [emptyRows 4], ...
                          'VariableTypes', {'double', 'double', 'double', 'cell'}, ...
                          'VariableNames', {'frameID', 'objectID', 'confidence', 'bbox'});
        trackingTable = [trackingTable; emptyTable];
    end
    
    % Add new entries
    rowIndices = (currentRows + 1):(currentRows + numEntries);
    trackingTable.frameID(rowIndices) = frameIdx;
    trackingTable.objectID(rowIndices) = ids;
    trackingTable.confidence(rowIndices) = scores;
    
    % Convert detections to cell array efficiently
    bboxCell = mat2cell(detections, ones(numEntries, 1), 4);
    trackingTable.bbox(rowIndices) = bboxCell;
    
    % Update actual row count
    trackingTable.Properties.UserData.actualRows = currentRows + numEntries;
end

% Vectorized IoU computation
function ious = bboxIoUVectorized(bboxes1, bboxes2)
    % Extract coordinates
    x1 = bboxes1(:, 1); y1 = bboxes1(:, 2); w1 = bboxes1(:, 3); h1 = bboxes1(:, 4);
    x2 = bboxes2(:, 1); y2 = bboxes2(:, 2); w2 = bboxes2(:, 3); h2 = bboxes2(:, 4);
    
    % Calculate intersection area (vectorized)
    xOverlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2));
    yOverlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2));
    intersectionArea = xOverlap .* yOverlap;
    
    % Calculate union area (vectorized)
    area1 = w1 .* h1;
    area2 = w2 .* h2;
    unionArea = area1 + area2 - intersectionArea;
    
    % Calculate IoU with division by zero protection
    ious = zeros(size(unionArea));
    validUnion = unionArea > 0;
    ious(validUnion) = intersectionArea(validUnion) ./ unionArea(validUnion);
end

% Helper function to compute IoU between two bounding boxes (kept for backward compatibility)
function iou = bboxIoU(bbox1, bbox2)
    ious = bboxIoUVectorized(bbox1, bbox2);
    iou = ious(1);
end

% Optimized Hungarian Algorithm with early termination
function [assignments, unassignedTracks, unassignedDetections] = ...
        assignDetectionsToTracks(cost, costThreshold)
    
    [m, n] = size(cost);
    
    % Early exit for empty cost matrix
    if m == 0 || n == 0
        assignments = [];
        unassignedTracks = 1:m;
        unassignedDetections = 1:n;
        return;
    end
    
    % OPTIMIZATION: Use greedy assignment for small problems
    if m * n <= 1000  % Threshold for greedy vs Hungarian
        assignment = greedyAssignment(cost);
    else
        assignment = hungarianAlgorithm(cost);
    end
    
    % Create assignment pairs where cost is below threshold
    validAssignments = [];
    for i = 1:m
        if assignment(i) > 0 && assignment(i) <= n && cost(i, assignment(i)) < costThreshold
            validAssignments = [validAssignments; i, assignment(i)];
        end
    end
    
    assignments = validAssignments;
    
    % Find unassigned tracks and detections
    if ~isempty(assignments)
        assignedTracks = assignments(:, 1);
        assignedDetections = assignments(:, 2);
    else
        assignedTracks = [];
        assignedDetections = [];
    end
    
    unassignedTracks = setdiff(1:m, assignedTracks)';
    unassignedDetections = setdiff(1:n, assignedDetections)';
end

% Fast greedy assignment for small problems
function assignment = greedyAssignment(costMatrix)
    [m, n] = size(costMatrix);
    assignment = zeros(m, 1);
    usedCols = false(n, 1);
    
    % Sort all costs and process in order
    [sortedCosts, indices] = sort(costMatrix(:));
    [rows, cols] = ind2sub([m, n], indices);
    
    for i = 1:length(sortedCosts)
        r = rows(i);
        c = cols(i);
        
        if assignment(r) == 0 && ~usedCols(c)
            assignment(r) = c;
            usedCols(c) = true;
        end
    end
end

% Optimized Hungarian Algorithm
function assignment = hungarianAlgorithm(costMatrix)
    [m, n] = size(costMatrix);
    
    % Make the matrix square by padding with large values if necessary
    maxCost = max(costMatrix(:));
    if isempty(maxCost) || ~isfinite(maxCost)
        maxCost = 1000;
    end
    
    if m < n
        costMatrix = [costMatrix; repmat(maxCost + 1, n - m, n)];
    elseif n < m
        costMatrix = [costMatrix, repmat(maxCost + 1, m, m - n)];
    end
    
    [rows, cols] = size(costMatrix);
    
    % Step 1: Subtract row minima (vectorized)
    rowMin = min(costMatrix, [], 2);
    costMatrix = costMatrix - rowMin;
    
    % Step 2: Subtract column minima (vectorized)
    colMin = min(costMatrix, [], 1);
    costMatrix = costMatrix - colMin;
    
    % Initialize assignment
    assignment = zeros(rows, 1);
    
    % Simple assignment finding (optimized for typical tracking scenarios)
    for maxIter = 1:rows
        % Find assignments
        assignment = findSimpleAssignment(costMatrix);
        
        % Check if we have enough assignments
        numAssigned = sum(assignment > 0);
        if numAssigned >= min(rows, cols)
            break;
        end
        
        % Create additional zeros
        costMatrix = createAdditionalZerosOptimized(costMatrix, assignment);
    end
    
    % Return only the first m assignments
    assignment = assignment(1:m);
    assignment(assignment > n) = 0;
end

function assignment = findSimpleAssignment(costMatrix)
    [rows, cols] = size(costMatrix);
    assignment = zeros(rows, 1);
    usedCols = false(cols, 1);
    
    % Find zeros and assign greedily
    [zeroRows, zeroCols] = find(costMatrix == 0);
    
    for i = 1:length(zeroRows)
        r = zeroRows(i);
        c = zeroCols(i);
        
        if assignment(r) == 0 && ~usedCols(c)
            assignment(r) = c;
            usedCols(c) = true;
        end
    end
end

function costMatrix = createAdditionalZerosOptimized(costMatrix, assignment)
    [rows, cols] = size(costMatrix);
    
    % Find covered rows and columns
    rowCovered = assignment > 0;
    colCovered = false(cols, 1);
    colCovered(assignment(assignment > 0)) = true;
    
    % Find minimum uncovered value
    uncoveredMask = ~rowCovered & ~colCovered';
    uncoveredElements = costMatrix(uncoveredMask);
    
    if isempty(uncoveredElements)
        return;
    end
    
    minUncovered = min(uncoveredElements);
    
    % Apply the Hungarian step (vectorized)
    costMatrix = costMatrix - minUncovered .* (~rowCovered);
    costMatrix = costMatrix + minUncovered .* (rowCovered & colCovered');
end