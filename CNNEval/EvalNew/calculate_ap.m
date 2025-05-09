function ap = calculate_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold)
% CALCULATE_AP Calculates Average Precision at specified IoU threshold
%
% If iou_threshold is not provided, defaults to 0.5 (AP50)
%
% INPUTS:
%   pred_boxes    - N x 4 matrix of predicted bounding boxes [x, y, h, w]
%   pred_scores   - N x 1 vector of confidence scores for predictions
%   gt_boxes      - M x 4 matrix of ground truth boxes [x, y, h, w]
%   iou_threshold - Scalar value between 0 and 1 for IoU threshold (default: 0.5)
%
% OUTPUT:
%   ap - Average Precision value at IoU threshold 0.5
%
% Examples:
%   % Calculate AP50 (default threshold)
%   pred_boxes = [10, 10, 50, 50; 20, 20, 40, 40; 200, 200, 30, 30];
%   pred_scores = [0.9; 0.8; 0.7];
%   gt_boxes = [12, 12, 48, 48; 210, 210, 25, 25];
%   ap50 = calculate_ap(pred_boxes, pred_scores, gt_boxes);
%
%   % Calculate AP75 (IoU threshold of 0.75)
%   ap75 = calculate_ap(pred_boxes, pred_scores, gt_boxes, 0.75);
%
%   % Calculate AP90 (IoU threshold of 0.9)
%   ap90 = calculate_ap(pred_boxes, pred_scores, gt_boxes, 0.9);

    % Set default IoU threshold if not provided
    if nargin < 4
        iou_threshold = 0.5;
    end
    
    % Check inputs
    assert(size(pred_boxes, 2) == 4, 'Predicted boxes should be Nx4 matrix');
    assert(size(gt_boxes, 2) == 4, 'Ground truth boxes should be Mx4 matrix');
    assert(length(pred_scores) == size(pred_boxes, 1), 'Number of scores should match number of predicted boxes');
    assert(iou_threshold > 0 && iou_threshold <= 1, 'IoU threshold should be between 0 and 1');

    % If no predictions or no ground truth, AP is 0
    if isempty(pred_boxes) || isempty(gt_boxes)
        ap = 0;
        return;
    end

    % Sort predictions by decreasing confidence
    [pred_scores, sort_idx] = sort(pred_scores, 'descend');
    pred_boxes = pred_boxes(sort_idx, :);
    
    % Initialize variables
    num_gt = size(gt_boxes, 1);
    num_pred = size(pred_boxes, 1);
    is_gt_matched = false(num_gt, 1);
    true_positives = zeros(num_pred, 1);
    false_positives = zeros(num_pred, 1);
    
    % For each prediction, determine if it's a TP or FP
    for i = 1:num_pred
        % Calculate IoU with all ground truth boxes
        ious = zeros(num_gt, 1);
        for j = 1:num_gt
            ious(j) = calculate_iou(pred_boxes(i,:), gt_boxes(j,:));
        end
        
        % Find the best matching ground truth box
        [max_iou, max_idx] = max(ious);
        
        % Check if it's a match (IoU >= threshold and not matched before)
        if max_iou >= iou_threshold && ~is_gt_matched(max_idx)
            true_positives(i) = 1;
            is_gt_matched(max_idx) = true;
        else
            false_positives(i) = 1;
        end
    end
    
    % Compute cumulative TP and FP counts
    cum_tp = cumsum(true_positives);
    cum_fp = cumsum(false_positives);
    
    % Calculate precision and recall at each detection
    precision = cum_tp ./ (cum_tp + cum_fp);
    recall = cum_tp / num_gt;
    
    % Ensure precision is non-decreasing from right to left (as per PASCAL VOC)
    for i = num_pred-1:-1:1
        precision(i) = max(precision(i), precision(i+1));
    end
    
    % Compute AP by calculating area under precision-recall curve
    recall_levels = [0, recall', 1];
    precision_levels = [0, precision', 0];
    
    ap = 0;
    for i = 1:length(recall_levels)-1
        ap = ap + (recall_levels(i+1) - recall_levels(i)) * precision_levels(i+1);
    end
end

function iou = calculate_iou(box1, box2)
% Helper function to calculate IoU between two boxes in [x, y, h, w] format
    
    % Convert [x, y, h, w] to [x1, y1, x2, y2] format
    box1_x1 = box1(1);
    box1_y1 = box1(2);
    box1_x2 = box1(1) + box1(3);
    box1_y2 = box1(2) + box1(4);
    
    box2_x1 = box2(1);
    box2_y1 = box2(2);
    box2_x2 = box2(1) + box2(3);
    box2_y2 = box2(2) + box2(4);
    
    % Calculate intersection area
    x_left = max(box1_x1, box2_x1);
    y_top = max(box1_y1, box2_y1);
    x_right = min(box1_x2, box2_x2);
    y_bottom = min(box1_y2, box2_y2);
    
    % Check if boxes intersect
    if x_right < x_left || y_bottom < y_top
        iou = 0;
        return;
    end
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top);
    
    % Calculate union area
    box1_area = box1(3) * box1(4);
    box2_area = box2(3) * box2(4);
    union_area = box1_area + box2_area - intersection_area;
    
    % Calculate IoU
    iou = intersection_area / union_area;
end