function loss = RPNClassLoss(Predictions, GTLabels)
    % cls_scores: HxWxNumAnchorsxB dlarray in SSCB format
    % gt_labels: HxWxNumAnchorsxB either dlarray or normal array
    
    % Reshape while preserving batch dimension
    orig_size = size(Predictions);
    batch_size = orig_size(4);
    
    % Reshape to 2D+batch for easier processing
    cls_scores_reshaped = reshape(Predictions, [], batch_size);
    gt_labels_reshaped = reshape(GTLabels, [], batch_size);
 
    
    % Create masks for positive and negative examples
    pos_mask = (gt_labels_reshaped == 1);
    neg_mask = (gt_labels_reshaped == 0);
    valid_mask = pos_mask | neg_mask;
    
    % Count positives and negatives per batch (without for loop)
    pos_count = sum(pos_mask, 1);
    neg_count = sum(neg_mask, 1);
    
    % Target negative factor (1:3 ratio)
    target_neg_factor = 3.0;
    
    % Calculate target negative counts (vectorized)
    % For batches with positives: min(pos_count * 3, neg_count)
    % For batches without positives: neg_count
    has_pos = (pos_count > 0);
    target_neg = has_pos .* min(pos_count * target_neg_factor, neg_count) + ...
                (~has_pos) .* neg_count;
    
    % Calculate scaling factors for negative examples (vectorized)
    neg_scale = zeros(1, batch_size, 'like', Predictions);
    valid_neg = (neg_count > 0);
    neg_scale(valid_neg) = target_neg(valid_neg) ./ neg_count(valid_neg);
    
    % Expand scaling factors to match dimensions of neg_mask
    neg_scale_expanded = repmat(neg_scale, [size(neg_mask, 1), 1]);
    
    % Apply scaling to negative examples
    neg_weights = neg_mask .* neg_scale_expanded;
    
    % Combined weights: 1.0 for positives, scaled weights for negatives
    sample_weights = pos_mask + neg_weights;
    
    % Apply numerical stability for log operations
    epsilon = 1e-12;
    cls_scores_stable = min(max(cls_scores_reshaped, epsilon), 1-epsilon);
    
    % Binary cross-entropy loss, element-wise
    bce_loss = -gt_labels_reshaped .* log(cls_scores_stable) - ...
               (1 - gt_labels_reshaped) .* log(1 - cls_scores_stable);
    
    % Apply sample weights and valid mask
    weighted_loss = bce_loss .* sample_weights .* valid_mask;
    
    % Sum losses and normalize (with epsilon for numerical stability)
    total_valid_weight = sum(sample_weights .* valid_mask, 1) + epsilon;
    batch_losses = sum(weighted_loss, 1) ./ total_valid_weight;
    
    % Average across batches
    loss = mean(batch_losses);
end