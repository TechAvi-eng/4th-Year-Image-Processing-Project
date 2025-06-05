
tic
target_boxes = [
    160, 180, 80, 120;   % Target 1
    320, 240, 64, 64;    % Target 2
    480, 360, 128, 96    % Target 3
];

anchor_sizes = [
    32, 32;   % Small square anchor
    64, 64;   % Medium square anchor
    128, 64;  % Wide anchor
    64, 128   % Tall anchor
];

feature_map_size = [33, 44];

[anchor_indices, ious, positions] = findClosestSizeAnchor(target_boxes, anchor_sizes, feature_map_size);

% Display results for each target box
for i = 1:size(target_boxes, 1)
    fprintf('Target box %d:\n', i);
    fprintf('  Best matching anchor size index: %d\n', anchor_indices(i));
    fprintf('  Position in feature map: [%d, %d]\n', positions(i,1), positions(i,2));
    fprintf('  IoU: %.4f\n\n', ious(i));
end

toc
function [closest_anchor_indices, max_ious, anchor_positions] = findClosestSizeAnchor(target_boxes, anchor_sizes, feature_map_size)
    % Find closest anchor boxes for multiple targets simultaneously
    %
    % Inputs:
    %   target_boxes: Mx4 matrix of [x, y, width, height] in original coordinates
    %   anchor_sizes: Nx2 matrix of anchor [width, height] pairs
    %   feature_map_size: [height, width] of the feature map (e.g., [14 14])
    %
    % Outputs:
    %   closest_anchor_indices: Mx1 vector of best matching anchor indices
    %   max_ious: Mx1 vector of best IoU values
    %   anchor_positions: Mx2 matrix of [row, col] positions in feature map

    % Setup constants
    stride = 16;  % Typical stride for MASK R-CNN
    img_height = feature_map_size(1) * stride;
    img_width = feature_map_size(2) * stride;
    num_targets = size(target_boxes, 1);
    num_anchors = size(anchor_sizes, 1);
    
    % Normalize target boxes to [0,1]
    target_boxes_norm = target_boxes ./ repmat([img_width, img_height, img_width, img_height], num_targets, 1);
    
    % Pre-compute all possible anchor centers
    [col_grid, row_grid] = meshgrid(1:feature_map_size(2), 1:feature_map_size(1));
    centers_x = (col_grid(:) - 0.5) / feature_map_size(2);  % Normalized x centers
    centers_y = (row_grid(:) - 0.5) / feature_map_size(1);  % Normalized y centers
    num_positions = length(centers_x);
    
    % Pre-compute normalized anchor sizes
    anchor_sizes_norm = anchor_sizes ./ repmat([img_width, img_height], num_anchors, 1);
    
    % Initialize output arrays
    max_ious = zeros(num_targets, 1);
    closest_anchor_indices = ones(num_targets, 1);
    anchor_positions = ones(num_targets, 2);
    
    % For each target box
    for i = 1:num_targets
        target_box = target_boxes_norm(i, :);
        best_iou = 0;
        
        % Create all possible anchors for this target (vectorized)
        all_ious = zeros(num_positions * num_anchors, 1);
        anchor_idx_map = zeros(num_positions * num_anchors, 1);
        position_map = zeros(num_positions * num_anchors, 2);
        
        idx = 1;
        % Generate all anchors at once
        for a = 1:num_anchors
            anchor_w = anchor_sizes_norm(a, 1);
            anchor_h = anchor_sizes_norm(a, 2);
            
            % Create anchor boxes for all positions
            anchor_boxes = [
                centers_x - anchor_w/2, ...
                centers_y - anchor_h/2, ...
                repmat(anchor_w, num_positions, 1), ...
                repmat(anchor_h, num_positions, 1)
            ];
            
            % Calculate IoUs for all positions of this anchor size
            ious = calculateBatchIoU(target_box, anchor_boxes);
            
            % Store results
            all_ious(idx:idx+num_positions-1) = ious;
            anchor_idx_map(idx:idx+num_positions-1) = a;
            position_map(idx:idx+num_positions-1, :) = [row_grid(:), col_grid(:)];
            
            idx = idx + num_positions;
        end
        
        % Find best match across all anchors and positions
        [max_iou, best_idx] = max(all_ious);
        
        % Store results for this target
        max_ious(i) = max_iou;
        closest_anchor_indices(i) = anchor_idx_map(best_idx);
        anchor_positions(i, :) = position_map(best_idx, :);
    end
end

function ious = calculateBatchIoU(box, boxes)
    % Vectorized IoU calculation between one box and many boxes
    % box: [x, y, width, height]
    % boxes: Nx4 matrix of [x, y, width, height]
    
    % Convert to [x1, y1, x2, y2] format
    box_coords = [
        box(1), ...
        box(2), ...
        box(1) + box(3), ...
        box(2) + box(4)
    ];
    
    boxes_coords = [
        boxes(:,1), ...
        boxes(:,2), ...
        boxes(:,1) + boxes(:,3), ...
        boxes(:,2) + boxes(:,4)
    ];
    
    % Calculate intersection coordinates
    x1_inter = max(box_coords(1), boxes_coords(:,1));
    y1_inter = max(box_coords(2), boxes_coords(:,2));
    x2_inter = min(box_coords(3), boxes_coords(:,3));
    y2_inter = min(box_coords(4), boxes_coords(:,4));
    
    % Calculate intersection areas
    inter_widths = max(0, x2_inter - x1_inter);
    inter_heights = max(0, y2_inter - y1_inter);
    intersections = inter_widths .* inter_heights;
    
    % Calculate areas
    box_area = box(3) * box(4);
    boxes_areas = boxes(:,3) .* boxes(:,4);
    
    % Calculate unions
    unions = box_area + boxes_areas - intersections;
    
    % Calculate IoUs
    ious = intersections ./ unions;
end