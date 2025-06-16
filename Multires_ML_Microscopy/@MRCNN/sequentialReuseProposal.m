function proposals = sequentialReuseProposal(obj, regressionBatch, scoresBatch, knownbboxes, numAdditionalProposals)
    arguments
        obj
        regressionBatch dlarray
        scoresBatch dlarray
        knownbboxes (:, 4) = []
        numAdditionalProposals (1,1) {mustBeInteger, mustBePositive} = max(size(knownbboxes, 1)*2, 100);
    end%2/3 of proposals are new by (with a minimum of 100) by default
% sequentialRegionProposal Region proposal functional layer reusing
% bounding boxes of the previous frame to make process more efficient
%
% proposals = regionProposal(obj, regressionBatch, scoresBatch, knownBBoxes)
% applies regressionBatch to the anchors to get a set of region proposals
% and reuses them with a desired amount of additional regions
% and refines it using selectStrongestBbox(NMS).
% regressionBatch is a h-by-w-by-numAnchors*4-by-batch dlarray(SSCB).
% scoresBatch is a h-by-w-by-numAnchors-by-batch dlarray(SSCB).
%
% The region proposal layer is part of the region proposal network (RPN)
% within Mask-RCNN. This layer outputs bounding boxes around potential
% objects in an image. The output proposals is a 5xM dlarray(SSCB) with
% each row containing the box proposal in the format [x1 y1 x2 y2 batch_idx]
% 
% These outputs are further refined by additional layers within Mask R-CNN
% to produce the final object detections and mask segmentations.

%catch empty known boxes--that is, no information from the previous frame--and do default proposal without foreknowledge
if isempty(knownbboxes)
    proposals = regionProposal(obj, regressionBatch, scoresBatch);
    return
end

featureSize = size(scoresBatch);
            
if isempty(obj.InputSize)
    imageSize = ceil(featureSize(1:2) .* fliplr(1./obj.ScaleFactor));
else
    imageSize = obj.InputSize;
end

% Get number of observations;
N = size(scoresBatch,4);
proposals = cell(N,1);

for i = 1:N
    cls = scoresBatch(:,:,:,i);
    reg = regressionBatch(:,:,:,i);
    
    [bboxes, scores] = proposeBoxes(cls, reg, obj.AnchorBoxes, obj.ScaleFactor, obj.RPNBoxStd, obj.RPNBoxMean);

    % Handle boxes outside of the image
    if strcmp(obj.ProposalsOutsideImage,'clip')
        bboxes = clipBBox(bboxes,imageSize);
    else
        H = imageSize(1);
        W = imageSize(2);
        outside =  bboxes(:,1) < 1 | bboxes(:,2) < 1 | (bboxes(:,1)+bboxes(:,3)-1) > W | (bboxes(:,2)+bboxes(:,4)-1) > H;
        bboxes(outside,:) = [];
        scores(outside) = [];
    end

    [bboxes, scores] = obj.BoxFilterFcn(bboxes, scores, obj.MinSize, obj.MaxSize);

    % remove low scoring proposals
    if obj.MinScore > 0
        lowScores = scores < obj.MinScore;
        bboxes(lowScores,:) = [];
        scores(lowScores,:) = [];
    end
    
    %Add known bboxes with maximum score to guarantee they are selected and
    %prioritized over overlapping proposals
    bboxes = cat(1,bboxes, knownbboxes);
    scores = cat(1, scores, ones([size(knownbboxes,1), 1]) * (max(scores)+1) );


    % PreNMS pick N strongest bboxes
    [~, ~, idx] = selectStrongestRegions(extractdata(bboxes),...
                                         extractdata(scores),...
                                         obj.NumStrongestRegionsBeforeProposalNMS); 
    bboxes = bboxes(idx,:);
    scores = scores(idx,:);

    % We don't need to trace the NMS operation for dlarray
    [~, ~, index] = selectStrongestBbox(extractdata(bboxes), extractdata(scores), ...
        'RatioType', 'Union', ...
        'OverlapThreshold', obj.OverlapThresholdRPN,...
        'NumStrongest', size(knownbboxes,1) + numAdditionalProposals); %add all known boxes and specified additonal amount

    bboxes = bboxes(index,:);
    
    bboxes = xywhToX1Y1X2Y2(bboxes);
    
    % Associate batch indices with each set of boxes.
    numBoxes = size(bboxes,1);
    proposals{i} = [bboxes repelem(i,numBoxes,1)];
end

proposals = single(vertcat(proposals{:}));
            
if(isempty(proposals))
    warning('Empty Proposals');
end

% convert to 5XM format
proposals = proposals';  

proposals = dlarray(proposals, 'SSCB');

end

function [bboxes, scores] = proposeBoxes(cls, reg, anchorBoxes, scaleFactor,RPNBoxStd, RPNBoxMean)
% Input cls are scores from the rpn cls conv layer 
%   size(cls): [H W 2*NumAnchors numObs] %%%%%%%%%%%%%%%%%%%%%%
% Input reg are scores from the rpn reg conv layer
%   size(reg): [H W 4*NumAnchors numObs]
%
% Output bboxes in [x y w h] format. In image space.

% anchor boxes in [width height]
aboxes = fliplr(anchorBoxes);

[fmapSizeY,fmapSizeX,numAnchors,~] = size(cls);
[y,x,k] = ndgrid(1:single(fmapSizeY), 1:single(fmapSizeX),1:size(aboxes,1));
x = x(:);
y = y(:);
k = k(:);
widthHeight = aboxes(k,:);

% Box centers in image space (spatial coordinates).
sx = 1/scaleFactor(1);
sy = 1/scaleFactor(2);
xCenter = x * sx + (1-sx)/2;
yCenter = y * sy + (1-sy)/2; 

halfWidthHeight = widthHeight / 2 ;

% top left in spatial coordinates
x1 = xCenter - halfWidthHeight(:,1);
y1 = yCenter - halfWidthHeight(:,2);

% Convert to pixel coordinates.
x1 = floor(x1 + 0.5);
y1 = floor(y1 + 0.5);

% anchor boxes as [x y w h]
bboxes = [x1 y1 widthHeight];

% Regression input is [H W 4*NumAnchors], where the 4
% box coordinates are consecutive elements. Reshape data to
% [H W 4 NumAnchors] so that we can gather the regression
% values based on the max score indices.
reg = reshape(reg, size(reg,1), size(reg,2), 4, []);

% permute reg so that it is [H W NumAnchors 4]
reg = permute(reg, [1 2 4 3]);

% reshape reg so that it is [H*W*NumAnchors 4]
reg = reshape(reg,[],4);

% Apply box target normalization
reg = (reg .* RPNBoxStd) + RPNBoxMean;

bboxes = fastRCNNObjectDetector.applyReg(bboxes, reg);

scores = reshape(cls(:,:,1:numAnchors,:),[],1);

[bboxes, scores] = fastRCNNObjectDetector.removeInvalidBoxesAndScores(bboxes, scores);

end

function boxes = xywhToX1Y1X2Y2(boxes)
    % Convert [x y w h] box to [x1 y1 x2 y2]. Input and output
    % boxes are in pixel coordinates. boxes is an M-by-4
    % matrix.
    boxes(:,3) = boxes(:,1) + boxes(:,3) - 1;
    boxes(:,4) = boxes(:,2) + boxes(:,4) - 1;
end

function clippedBBox = clipBBox(bbox, imgSize)
    % bounding boxes are returned as doubles
    clippedBBox  = single(bbox);

    % The original bounding boxes all overlap the image. Therefore, a check to
    % remove non-overlapping boxes is not required.

    % Get coordinates of upper-left (x1,y1) and bottom-right (x2,y2) corners. 
    x1 = clippedBBox(:,1);
    y1 = clippedBBox(:,2);

    x2 = clippedBBox(:,1) + clippedBBox(:,3) - 1;
    y2 = clippedBBox(:,2) + clippedBBox(:,4) - 1;

    x1(x1 < 1) = 1;
    y1(y1 < 1) = 1;

    x2(x2 > imgSize(2)) = imgSize(2);
    y2(y2 > imgSize(1)) = imgSize(1);

    clippedBBox = [x1 y1 x2-x1+1 y2-y1+1];
end

function [bboxes, scores, idx] = selectStrongestRegions(bboxes, scores, N)
    if ~isinf(N)
        [~, idx] = sort(scores, 'descend');
        topN = min(N, numel(scores));
        idx = idx(1:topN);
        bboxes = bboxes(idx,:);
        scores = scores(idx,:);
    end
end


