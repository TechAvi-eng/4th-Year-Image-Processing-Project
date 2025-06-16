

function outFeatures = roiAlignPooling (obj, X, boxes, poolSize)
% roiAlignPooling Non-quantized ROI pooling layer for Mask-CNN.
%
% roiAlignPooling performs roialign operation on input X which is a
% h-by-w-by-numFeat-by-numBatch dlarray(SSCB) using boxes -
% 5-by-numProposals dlarray (SSCB). Each column of boxes holds a box
% proposal in [x1 y1 x2 y2 batch_idx] format.
% The output is a pooled featuremap, which is a dlarray of size
% poolSize(1)-by-poolSize(2)-by-numFeat-numProposals and format SSCB.
    
    % Scale the boxes to feature coordinates

    boxes(1:4, :, :, :) = ((boxes(1:4,:, :, :)-1 ) .* obj.ScaleFactor(1)) + 0.5;

    outFeatures = roialign(X, squeeze(boxes), poolSize, 'ROIScale', 1);
end