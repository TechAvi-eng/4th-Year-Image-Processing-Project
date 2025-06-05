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
    
    % Initialize tracks if this is the first frame
    if isempty(tracks)
        tracks = struct('id', {}, 'bbox', {}, 'score', {}, 'age', {}, ...
                      'totalVisibleCount', {}, 'consecutiveInvisibleCount', {});
        nextId = 1;
    else
        % Get the maximum ID from existing tracks
        ids = [tracks.id];
        if isempty(ids)
            nextId = 1;
        else
            nextId = max(ids) + 1;
        end
    end
    
    % Initialize tracking table if needed
    if isempty(trackingTable)
        trackingTable = table('Size', [0 4], ...
                          'VariableTypes', {'double', 'double', 'double', 'cell'}, ...
                          'VariableNames', {'frameID', 'objectID', 'confidence', 'bbox'});
    end
    
    % Skip association if no current tracks
    if isempty(tracks)
        % Create a new track for each detection
        for i = 1:size(detections, 1)
            newTrack = struct(...
                'id', nextId, ...
                'bbox', detections(i, :), ...
                'score', scores(i), ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            
            tracks(end+1) = newTrack;
            
            % Add to tracking table with the actual confidence score
            newRow = {frameIdx, nextId, scores(i), {detections(i, :)}};
            trackingTable = [trackingTable; newRow];
            
            nextId = nextId + 1;
        end
        return;
    end
    
    % Calculate combined cost metric between each detection and track
    numDetections = size(detections, 1);
    numTracks = length(tracks);
    costMatrix = zeros(numTracks, numDetections);
    
    for i = 1:numTracks
        track_bbox = tracks(i).bbox;
        track_width = track_bbox(3);
        track_height = track_bbox(4);
        track_size = track_width * track_height;
        track_aspect = track_width / track_height;
        
        for j = 1:numDetections
            det_bbox = detections(j, :);
            det_width = det_bbox(3);
            det_height = det_bbox(4);
            det_size = det_width * det_height;
            det_aspect = det_width / det_height;
            
            % Calculate IoU component (lower is better)
            iou = bboxIoU(track_bbox, det_bbox);
            iou_cost = 1 - iou;
            
            % Calculate size difference component (normalized between 0 and 1)
            % Using relative size difference: |s1-s2|/max(s1,s2)
            size_diff = abs(track_size - det_size) / max(track_size, det_size);
            
            % Calculate aspect ratio difference component (normalized between 0 and 1)
            % Using a bounded metric that approaches 1 as ratios diverge
            aspect_diff = abs(track_aspect - det_aspect) / (1 + abs(track_aspect - det_aspect));
            
            % Combine the components with weights
            costMatrix(i, j) = args.IoUWeight * iou_cost + ...
                               args.SizeWeight * size_diff + ...
                               args.AspectRatioWeight * aspect_diff;
        end
    end
    
    % Use assignment algorithm
    [assignments, unassignedTracks, unassignedDetections] = ...
        assignDetectionsToTracks(costMatrix, iouThreshold);
    
    % Update assigned tracks
    for i = 1:size(assignments, 1)
        trackIdx = assignments(i, 1);
        detectionIdx = assignments(i, 2);
        
        tracks(trackIdx).bbox = detections(detectionIdx, :);
        tracks(trackIdx).score = scores(detectionIdx);  % Update score
        tracks(trackIdx).age = tracks(trackIdx).age + 1;
        tracks(trackIdx).totalVisibleCount = tracks(trackIdx).totalVisibleCount + 1;
        tracks(trackIdx).consecutiveInvisibleCount = 0;
        
        % Add to tracking table with the actual confidence score
        newRow = {frameIdx, tracks(trackIdx).id, scores(detectionIdx), {detections(detectionIdx, :)}};
        trackingTable = [trackingTable; newRow];
    end
    
    % Update unassigned tracks
    for i = 1:length(unassignedTracks)
        idx = unassignedTracks(i);
        tracks(idx).age = tracks(idx).age + 1;
        tracks(idx).consecutiveInvisibleCount = tracks(idx).consecutiveInvisibleCount + 1;
        
        % Optionally predict new position based on previous motion
        % (not implemented here, but you could add motion prediction)
    end
    
    % Create new tracks for unassigned detections
    for i = 1:length(unassignedDetections)
        detectionIdx = unassignedDetections(i);
        newTrack = struct(...
            'id', nextId, ...
            'bbox', detections(detectionIdx, :), ...
            'score', scores(detectionIdx), ...  % Store score
            'age', 1, ...
            'totalVisibleCount', 1, ...
            'consecutiveInvisibleCount', 0);
        
        tracks(end+1) = newTrack;
        
        % Add to tracking table with the actual confidence score
        newRow = {frameIdx, nextId, scores(detectionIdx), {detections(detectionIdx, :)}};
        trackingTable = [trackingTable; newRow];
        
        nextId = nextId + 1;
    end
    
    % Remove dead tracks that have been invisible for too long
    isDead = [tracks.consecutiveInvisibleCount] >= invisibleForTooLong;
    tracks = tracks(~isDead);
end

% Helper function to compute IoU between two bounding boxes
function iou = bboxIoU(bbox1, bbox2)
    % Extract coordinates
    x1 = bbox1(1); y1 = bbox1(2); w1 = bbox1(3); h1 = bbox1(4);
    x2 = bbox2(1); y2 = bbox2(2); w2 = bbox2(3); h2 = bbox2(4);
    
    % Calculate intersection area
    xOverlap = max(0, min(x1+w1, x2+w2) - max(x1, x2));
    yOverlap = max(0, min(y1+h1, y2+h2) - max(y1, y2));
    intersectionArea = xOverlap * yOverlap;
    
    % Calculate union area
    area1 = w1 * h1;
    area2 = w2 * h2;
    unionArea = area1 + area2 - intersectionArea;
    
    % Calculate IoU
    iou = intersectionArea / unionArea;
end

% Assignment function (simplified version - you can replace with MATLAB's built-in assignDetectionsToTracks)
function [assignments, unassignedTracks, unassignedDetections] = ...
        assignDetectionsToTracks(cost, costThreshold)
    % Initialize outputs
    assignments = [];
    unassignedTracks = 1:size(cost, 1);
    unassignedDetections = 1:size(cost, 2);
    
    if isempty(cost)
        return;
    end
    
    % Find assignments with cost below threshold
    [m, n] = size(cost);
    
    % Use the Hungarian algorithm
    % Note: You could use MATLAB's built-in assignDetectionsToTracks if you have the Computer Vision Toolbox
    [assignment, ~] = munkres(cost);
    
    % Create assignment pairs where cost is below threshold
    validIdx = false(m, 1);
    for i = 1:m
        if assignment(i) > 0 && cost(i, assignment(i)) < costThreshold
            validIdx(i) = true;
        end
    end
    
    assignments = [find(validIdx), assignment(validIdx)];
    
    % Find unassigned tracks and detections
    assignedTracks = assignments(:, 1);
    unassignedTracks = setdiff(1:m, assignedTracks)';
    
    assignedDetections = assignments(:, 2);
    unassignedDetections = setdiff(1:n, assignedDetections)';
end