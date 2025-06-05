function loss = featureDiversityLoss(features)
    % This function computes a diversity loss while maintaining dlarray type
    % Input: features - dlarray in 'SSCB' format (height, width, channels, batch)
    % Output: loss - dlarray scalar representing the diversity loss
    
    
    % Get dimensions
    [H, W, C, B] = size(features);
    totalLoss = 0;
    
    % Process each batch sample separately
    for b = 1:B
        % Extract features for current batch item (still a dlarray)
        feat_batch = features(:,:,:,b);
        
        % Reshape to [C × (H*W)] while preserving dlarray
        feat_flat = reshape(feat_batch, H*W, C);
        feat_flat = permute(feat_flat, [2, 1]); % Now C×(H*W)
        
        % Compute L2 norm for each feature map (channel)
        channel_norms = sqrt(sum(feat_flat.^2, 2) + 1e-8);
        
        % Normalize each feature map to unit length while preserving dlarray
        feat_norm = feat_flat ./ channel_norms;
        
        % Calculate cosine similarity matrix (C×C matrix)
        sim_matrix = feat_norm * permute(feat_norm, [2, 1]);
        
        % Create diagonal mask as dlarray
        diag_mask = dlarray(1 - eye(C, 'like', gather(extractdata(sim_matrix))), 'SS');
        
        % Apply mask (element-wise multiply) to zero out self-similarity
        masked_sim = sim_matrix .* diag_mask;
        
        % Compute squared similarities
        squared_sim = masked_sim.^2;
        
        % Sum all squared similarities and normalize
        num_elements = C * (C - 1); % Number of off-diagonal elements
        batch_loss = sum(squared_sim, 'all') / num_elements;
        
        totalLoss = totalLoss + batch_loss;
    end
    
    % Average over batch
    loss = totalLoss / B;
end