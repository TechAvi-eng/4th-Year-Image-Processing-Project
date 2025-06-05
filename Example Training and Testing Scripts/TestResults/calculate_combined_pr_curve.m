function [precision, recall, ap] = calculate_combined_pr_curve(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold)
% CALCULATE_COMBINED_PR_CURVE Calculates precision-recall curve across multiple images/sets
%
% INPUTS:
%   all_pred_boxes  - Cell array where each cell contains Nx4 matrix of predicted boxes [x,y,h,w]
%   all_pred_scores - Cell array where each cell contains Nx1 vector of confidence scores
%   all_gt_boxes    - Cell array where each cell contains Mx4 matrix of ground truth boxes [x,y,h,w]
%   iou_threshold   - Scalar value between 0 and 1 for IoU threshold (default: 0.5)
%
% OUTPUTS:
%   precision - Vector of precision values at each detection threshold
%   recall    - Vector of recall values at each detection threshold
%   ap        - Average Precision value at specified IoU threshold
%
% Example:
%   % Three separate images/sets of predictions and ground truths
%   pred_boxes_1 = [10, 10, 50, 50; 20, 20, 40, 40];
%   pred_scores_1 = [0.9; 0.8];
%   gt_boxes_1 = [12, 12, 48, 48];
%
%   pred_boxes_2 = [200, 200, 30, 30; 150, 150, 40, 40];
%   pred_scores_2 = [0.7; 0.6];
%   gt_boxes_2 = [210, 210, 25, 25];
%
%   pred_boxes_3 = [300, 300, 60, 60; 400, 400, 50, 50];
%   pred_scores_3 = [0.95; 0.85];
%   gt_boxes_3 = [310, 310, 55, 55; 405, 405, 45, 45];
%
%   % Combine into cell arrays
%   all_pred_boxes = {pred_boxes_1, pred_boxes_2, pred_boxes_3};
%   all_pred_scores = {pred_scores_1, pred_scores_2, pred_scores_3};
%   all_gt_boxes = {gt_boxes_1, gt_boxes_2, gt_boxes_3};
%
%   % Calculate combined PR curve
%   [precision, recall, ap] = calculate_combined_pr_curve(all_pred_boxes, all_pred_scores, all_gt_boxes);

    % Set default IoU threshold if not provided
    if nargin < 4
        iou_threshold = 0.5;
    end
    
    % Check inputs
    assert(iscell(all_pred_boxes), 'all_pred_boxes should be a cell array');
    assert(iscell(all_pred_scores), 'all_pred_scores should be a cell array');
    assert(iscell(all_gt_boxes), 'all_gt_boxes should be a cell array');
    assert(length(all_pred_boxes) == length(all_pred_scores), 'all_pred_boxes and all_pred_scores should have the same length');
    assert(length(all_pred_boxes) == length(all_gt_boxes), 'all_pred_boxes and all_gt_boxes should have the same length');
    assert(iou_threshold > 0 && iou_threshold <= 1, 'IoU threshold should be between 0 and 1');
    
    % Count total number of ground truths
    total_gt_count = 0;
    for i = 1:length(all_gt_boxes)
        if ~isempty(all_gt_boxes{i})
            total_gt_count = total_gt_count + size(all_gt_boxes{i}, 1);
        end
    end
    
    % Handle empty case
    if total_gt_count == 0
        precision = 0;
        recall = 0;
        ap = 0;
        % Plot empty PR curve
        figure;
        plot(0, 0, 'r-');
        title(['Combined Precision-Recall Curve (IoU = ' num2str(iou_threshold) ', AP = 0)']);
        xlabel('Recall');
        ylabel('Precision');
        axis([0 1 0 1]);
        grid on;
        return;
    end
    
    % Collect all predictions and their results
    all_image_ids = [];     % Image ID for each detection
    all_confidences = [];   % Confidence score for each detection
    all_is_tp = [];         % Whether each detection is a true positive (1) or false positive (0)
    
    % Process each image/set
    for img_id = 1:length(all_pred_boxes)
        pred_boxes = all_pred_boxes{img_id};
        pred_scores = all_pred_scores{img_id};
        gt_boxes = all_gt_boxes{img_id};
        
        % Skip if no predictions or ground truths
        if isempty(pred_boxes) || isempty(gt_boxes)
            continue;
        end
        
        % Ensure correct format
        assert(size(pred_boxes, 2) == 4, 'Predicted boxes should be Nx4 matrix');
        assert(size(gt_boxes, 2) == 4, 'Ground truth boxes should be Mx4 matrix');
        assert(length(pred_scores) == size(pred_boxes, 1), 'Number of scores should match number of predicted boxes');
        
        num_pred = size(pred_boxes, 1);
        num_gt = size(gt_boxes, 1);
        
        % Keep track of which GT boxes have been matched
        is_gt_matched = false(num_gt, 1);
        
        % Store image ID, confidence scores and whether each detection is TP or FP
        all_image_ids = [all_image_ids; repmat(img_id, num_pred, 1)];
        all_confidences = [all_confidences; pred_scores];
        
        % Evaluate each prediction
        for pred_idx = 1:num_pred
            % Calculate IoU with all ground truth boxes
            ious = zeros(num_gt, 1);
            for gt_idx = 1:num_gt
                ious(gt_idx) = calculate_iou(pred_boxes(pred_idx,:), gt_boxes(gt_idx,:));
            end
            
            % Find the best matching ground truth box
            [max_iou, max_idx] = max(ious);
            
            % Check if it's a match (IoU >= threshold and not matched before)
            if max_iou >= iou_threshold && ~is_gt_matched(max_idx)
                all_is_tp = [all_is_tp; 1];  % True positive
                is_gt_matched(max_idx) = true;
            else
                all_is_tp = [all_is_tp; 0];  % False positive
            end
        end
    end
    
    % If no valid predictions were found
    if isempty(all_confidences)
        precision = 0;
        recall = 0;
        ap = 0;
        % Plot empty PR curve
        figure;
        plot(0, 0, 'r-');
        title(['Combined Precision-Recall Curve (IoU = ' num2str(iou_threshold) ', AP = 0)']);
        xlabel('Recall');
        ylabel('Precision');
        axis([0 1 0 1]);
        grid on;
        return;
    end
    
    % Sort all detections by decreasing confidence
    [sorted_confidences, sort_idx] = sort(all_confidences, 'descend');
    sorted_is_tp = all_is_tp(sort_idx);
    
    % Compute cumulative TP and FP counts
    cum_tp = cumsum(sorted_is_tp);
    cum_fp = cumsum(~sorted_is_tp);
    
    % Calculate precision and recall at each detection
    precision = cum_tp ./ (cum_tp + cum_fp);
    recall = cum_tp / total_gt_count;
    %recall = rescale(recall)*0.95;
    % Add beginning and end points for plotting a complete curve
    precision_for_plot = [1; precision; 0];
    recall_for_plot = [0; recall; 1];
    
    % Compute AP using the interpolated PASCAL VOC method
    precision_interp = precision;
    num_pred = length(precision);
    
    % Ensure precision is non-decreasing from right to left (as per PASCAL VOC)
    for i = num_pred-1:-1:1
        precision_interp(i) = max(precision_interp(i), precision_interp(i+1));
    end
    
    % Compute AP by calculating area under precision-recall curve
    recall_levels = [0, recall', 1];
    precision_levels = [0, precision_interp', 0];
    
    ap = 0;
    for i = 1:length(recall_levels)-1
        ap = ap + (recall_levels(i+1) - recall_levels(i)) * precision_levels(i+1);
    end
    
    % Plot the precision-recall curve
    figure;
    
    % Plot the raw PR curve
    plot(recall_for_plot, precision_for_plot, 'b-', 'LineWidth', 1);
    hold on;
    
    % Plot the interpolated precision used for AP calculation
    plot([0; recall; 1], [1; precision_interp; 0], 'r--', 'LineWidth', 1);
    
    % Add AP value to title
    title(['Combined Precision-Recall Curve (IoU = ' num2str(iou_threshold) ', AP = ' num2str(ap, '%.4f') ')']);
    xlabel('Recall');
    ylabel('Precision');
    axis([0 1 0 1]); % Set axis limits
    grid on;
    legend('PR curve', 'Interpolated precision', 'Location', 'southwest');
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