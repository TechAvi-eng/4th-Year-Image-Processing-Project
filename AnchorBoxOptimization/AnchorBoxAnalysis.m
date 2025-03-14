clear
clc
close all
load("all_bbox_data.mat")

dims = bbox(:, 3:4); %height and width data from all anchor boxes

%%
% k-means clustering to find optimal anchor box setup at the given
% magnification
k=9;
[idx centroids] = kmeans(dims, k, 'Replicates',10,'MaxIter',5000);
anchor_widths = centroids(:,1);
anchor_heights = centroids(:,2);

%%
% Calculate aspect ratios
aspect_ratios = anchor_widths ./ anchor_heights;

% Sort by area for better interpretation
areas = anchor_widths .* anchor_heights;
[~, area_idx] = sort(areas);
sorted_anchors = [anchor_widths(area_idx), anchor_heights(area_idx), aspect_ratios(area_idx)];

% Display results
disp('Optimized anchors (width, height, aspect ratio):');
disp(sorted_anchors);

% Visualize anchor boxes
figure;
cs = lines(length(anchor_heights));

%%
for i = 1:length(anchor_heights)
    rectangle('Position', [0, 0, anchor_widths(i), anchor_heights(i)], ...
              'EdgeColor', cs(i,:), LineWidth=2);
    hold on;
end
axis equal;
title('Optimized Anchor Boxes');
xlim([0 71])
ylim([0 71])


%%
% test with own aspect ratios of from the previous one if this section is
% not run

abs = generateAnchorBoxes([14 21 32 47 71], [0.75 1 1.5]);


% optimal configuration that is consistent with sizes and aspect ratios
abs = [14 14; 14 21; 21 14;...
    21 21; 21 32; 32 21;...
    32 32; 47 32; 32 47;...
    47 47; 71 47; 47 71;...
    71 71];

 % ABs = [20 20; 20 40; 40 20]*2; %defaults
 % abs = [ABs; ABs*2; ABs*4; ABs*8];

anchor_widths = abs(:,1);
anchor_heights = abs(:,2);

%%
% Calculate IoU between ground truth boxes and optimized anchors
max_ious = zeros(size(dims, 1), 1);

for i = 1:size(dims, 1)
    gt_width = dims(i, 1);
    gt_height = dims(i, 2);
    
    for j = 1:size(anchor_heights,1)
        % Calculate IoU between ground truth and anchor
        % (simplified version - assumes boxes are centered)
        min_w = min(gt_width, anchor_widths(j));
        min_h = min(gt_height, anchor_heights(j));
        
        intersection = min_w * min_h;
        union = (gt_width * gt_height) + (anchor_widths(j) * anchor_heights(j)) - intersection;
        
        iou = intersection / union;
        max_ious(i) = max(max_ious(i), iou);
    end
end

%max_kmc = max_ious;
% Show statistics
mean_iou = mean(max_ious);
disp(['Mean max IoU: ', num2str(mean_iou)]);
histogram(max_ious);
title('Max IoU Distribution');

%%
%histogram(max_unopt, 'edgecolor','none');
hold on

histogram(max_kmc, 'edgecolor','none');
histogram(max_ious, 'edgecolor','none');



title('Max IoU Distribution', Interpreter='latex');
fontname("CMU Serif")
fontsize(12, "points")
%legend("K-Means Clustering Results", "Optimized Anchor Boxes", Location='northwest',Interpreter='latex');

legend("Unoptimized Boxes", "K-Means Clustering Results", "Optimized Anchor Boxes", Location='northwest',Interpreter='latex');


%%
function anchorBoxes = generateAnchorBoxes(scales, aspectRatios)
    % generateAnchorBoxes - Create anchor boxes ensuring one dimension matches a scale
    %
    % Inputs:
    %   scales - Vector of anchor box scales (e.g., [10, 20])
    %   aspectRatios - Vector of aspect ratios (e.g., [0.5, 1, 2])
    %
    % Output:
    %   anchorBoxes - Nx2 matrix where each row is [width, height]

    % Get min and max scales from the input
    minScale = min(scales);
    maxScale = max(scales);

    % Initialize list to store boxes
    anchorList = [];

    % Generate all combinations
    for i = 1:length(scales)
        scale = scales(i);
        for j = 1:length(aspectRatios)
            ratio = aspectRatios(j);
            
            % Option 1: Keep scale as width, adjust height
            width1 = scale;
            height1 = scale * ratio;
            if height1 >= minScale && height1 <= maxScale  % Ensure within scale limits
                anchorList = [anchorList; width1, height1];
            end

            % Option 2: Keep scale as height, adjust width
            width2 = scale / ratio;
            height2 = scale;
            if width2 >= minScale && width2 <= maxScale  % Ensure within scale limits
                anchorList = [anchorList; width2, height2];
            end
        end
    end

    % Remove duplicates and near-duplicates
    anchorBoxes = unique(round(anchorList, 3), 'rows'); % Round to 3 decimal places to remove near-duplicates

    % Show warning if some combinations were removed
    numFiltered = size(anchorList, 1) - size(anchorBoxes, 1);
    if numFiltered > 0
        fprintf('Warning: %d near-duplicate or out-of-range anchor boxes were removed.\n', numFiltered);
    end
end
