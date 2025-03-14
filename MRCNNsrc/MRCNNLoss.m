classdef MRCNNLoss < images.dltrain.internal.Loss
% This class defines the loss for maskrcnn object training

% Copyright 2021-2023 The MathWorks, Inc.
    properties (Access=private)
        params
    end

    properties
        MetricNames = ["Loss","RPNClass","RPNReg","Class", "Reg", "MaskLoss", "CountLoss"]
    end



    methods

        function obj = MRCNNLoss(params)
            obj.params = params;
        end

        function [loss, lossData] = lossFcn (obj, YRPNClass, YRPNRegDeltas, proposal, YRCNNClass, YRCNNReg, YMask, gTruthBoxes, gTruthLabels, gTruthMasks)
        
        % Proposals are 5XNumProposals (Due to batch restrictions from custom RPL layer)
        proposals = gather(extractdata(proposal));
        
        % Convert proposals to numProposals x 5 (as expected by the rest of post processing code)
        proposals =proposals';
        
        proposals(:,1:4) = vision.internal.cnn.maskrcnn.boxUtils.x1y1x2y2ToXYWH(proposals(:,1:4));
        
        numImagesInBatch = size(gTruthBoxes,1);
        %Convert numProposalsx5 Proposals to numImagesInBatchx1 (Group by image index)
        proposals = vision.internal.cnn.maskrcnn.groupProposalsByImageIndex(proposals, numImagesInBatch);
        
        
        % Generate RCNN response targets
        %--------------------------------
        
        % Step 1: Match ground truth boxes to proposals                                       
        [assignment, positiveIndex, negativeIndex] = vision.internal.cnn.maskrcnn.bboxMatchAndAssign(...
                                                                proposals, gTruthBoxes,...
                                                                obj.params.PositiveOverlapRange, obj.params.NegativeOverlapRange,...
                                                                obj.params.ForcedPositiveProposals);
        
                                                            
        % Step 2: Calcuate regression targets as (dx, dy, log(dw), log(dh))
        regressionTargets = vision.internal.cnn.maskrcnn.generateRegressionTargets(gTruthBoxes, proposals,...
                                                                assignment, positiveIndex,...
                                                                obj.params.NumClasses);
         
        classNames = categories(gTruthLabels{1});
        
        % Step 3: Assign groundtrutrh labels to proposals
        classificationTargets = vision.internal.cnn.maskrcnn.generateClassificationTargets (gTruthLabels, assignment,...
                                                     positiveIndex, negativeIndex,...
                                                     classNames, obj.params.BackgroundClass);
                                                 
        % Step 4: Calculate instance weights
        instanceWeightsReg = vision.internal.cnn.maskrcnn.regressionResponseInstanceWeights (classificationTargets, obj.params.BackgroundClass);
         
        % Step 5: Generate mask targets
         
        % Crop and resize the instances based on proposal bboxes and network output size
        maskOutputSize = obj.params.MaskOutputSize;
        croppedMasks = vision.internal.cnn.maskrcnn.cropandResizeMasks (gTruthMasks, gTruthBoxes, maskOutputSize);
         
        % Generate mask targets
        maskTargets = vision.internal.cnn.maskrcnn.generateMaskTargets(croppedMasks, assignment, classificationTargets, obj.params);
        
        % Stage 2 (RCNN) Loss
        % --------------------
        
        % *Classification loss*
        classificationTargets = cat(1, classificationTargets{:})';
        % onehotencode labels                       
        classificationTargets = onehotencode(classificationTargets,1);
        classificationTargets(isnan(classificationTargets)) = 0;
        
        classificationTargets = reshape(classificationTargets ,1, 1, size(YRCNNClass,3),[]);
        LossRCNNClass = vision.internal.cnn.maskrcnn.CrossEntropy(YRCNNClass, classificationTargets);
         
        % *Weighted regression loss*
        regressionTargets = cat(1,regressionTargets{:})';
        instanceWeightsReg = cat(1, instanceWeightsReg{:})';
        
        regressionTargets = reshape(regressionTargets, 1, 1, size(YRCNNReg,3),[]);
        instanceWeightsReg = reshape(instanceWeightsReg, 1, 1, size(YRCNNReg,3),[]);
        LossRCNNReg = vision.internal.cnn.maskrcnn.smoothL1(YRCNNReg, single(regressionTargets), single(instanceWeightsReg));

        % Mask Loss (Weighted cross entropy)
        maskTargets= cat(4,maskTargets{:});
        positiveIndex = cat(1,positiveIndex{:});
        LossRCNNMask = vision.internal.cnn.maskrcnn.SpatialCrossEntropy(YMask, single(maskTargets), positiveIndex);
         
        % gtNum = size(gTruthLabels{1}, 1);
        % posPredicts=max(extractdata(squeeze(YRCNNClass))-0.5, 0);
        % numPosPredict=nnz(posPredicts(1:end-1, :));
        % LossRCNNNum = abs(gtNum - numPosPredict)/(gtNum + numPosPredict);
        
        % Total Stage 2 loss
        LossRCNN = LossRCNNReg + LossRCNNClass + LossRCNNMask;
        
         
        % Generate RCNN response targets
        %--------------------------------
        featureSize = size(YRPNRegDeltas);
        imageSize = obj.params.ImageSize;
        [RPNRegressionTargets, RPNRegWeights, assignedLabelsRPN] = vision.internal.cnn.maskrcnn.rpnRegressionResponse(featureSize, gTruthBoxes, imageSize, obj.params);
        
        RPNClassificationTargets = onehotencode(assignedLabelsRPN, 3);
        % detetron2 uses ony foreground class for classification.
        RPNClassificationTargets(:,:,2,:) = [];
        RPNClassificationTargets  = reshape(RPNClassificationTargets, featureSize(1), featureSize(2), [], numImagesInBatch );
        RPNClassificationTargets(isnan(RPNClassificationTargets)) = 0;
        
        
        % Stage 1 (RPN) Loss
        % --------------------
        YRPNClass = sigmoid(YRPNClass);

        LossRPNClass = NegativeMining(YRPNClass, RPNClassificationTargets, 3); 
        %LossRPNClass = vision.internal.cnn.maskrcnn.CrossEntropy(YRPNClass, RPNClassificationTargets);

        LossRPNReg = vision.internal.cnn.maskrcnn.smoothL1(YRPNRegDeltas, RPNRegressionTargets, RPNRegWeights);

        LossRPN = LossRPNClass + LossRPNReg;
        
        numGT  = cellfun(@(x) size(x, 1), gTruthLabels);

        numProp = cellfun(@(x) size(x, 1), proposals);
        numPosPred = cumsum(squeeze(extractdata(YRCNNClass(:,:,1,:)))>0.5);
        numPosPred = numPosPred(cumsum(numProp));

        LossObjCount = smape(numGT, numPosPred);

        
        % Total Loss
        %------------
        loss = LossRCNN + LossRPN;
    
        lossData.Loss = loss;
        lossData.RPNClass = LossRPNClass;
        lossData.RPNReg = LossRPNReg;
        lossData.RPNLoss = LossRPN;
        lossData.Class = LossRCNNClass;
        lossData.Reg = LossRCNNReg;
        lossData.MaskLoss = LossRCNNMask;
        lossData.CountLoss = LossObjCount;

        end
    end

end


function smape_value = smape(actual, predicted)
    % SMAPE - Symmetric Mean Absolute Percentage Error
    
    % Calculate the absolute difference between actual and predicted
    abs_diff = abs(actual - predicted);
    
    % Calculate the average of absolute actual and predicted values
    abs_avg = (abs(actual) + abs(predicted)) / 2;
    
    % Find indices where the denominator is not zero
    valid_indices = abs_avg ~= 0;
    
    % Return 0 if there are no valid indices
    if sum(valid_indices) == 0
        smape_value = 0;
    else
        % Calculate SMAPE only for valid indices
        smape_valid = abs_diff(valid_indices) ./ abs_avg(valid_indices);
        
        % Take the mean
        smape_value = mean(smape_valid);
    end
end