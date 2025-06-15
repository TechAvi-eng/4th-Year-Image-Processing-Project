classdef MRCNNLoss < images.dltrain.internal.Loss
% This class defines the loss for maskrcnn object training

% Copyright 2021-2023 The MathWorks, Inc.
    properties (Access=public)
        params
    end

    properties
        MetricNames = ["Loss","RPNClass","RPNReg","Class1","Class2","Class3","Class4", "Reg1","Reg2","Reg3","Reg4", "MaskLoss"]
    end


    

    methods

        function obj = MRCNNLoss(params)
            obj.params = params;
        end

        function [loss, lossData] = lossFcn (obj, YRPNClass,CCScore1, CCScore2,CCScore3, YRPNRegDeltas, CCRegDeltas1, CCRegDeltas2,CCRegDeltas3, proposal1, proposal2,proposal3, proposal4, YRCNNClass, YRCNNReg, YMask, gTruthBoxes, gTruthLabels, gTruthMasks)
        
        
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
        LossRPNClass = RPNClassLoss(YRPNClass, RPNClassificationTargets);
        LossRPNReg = vision.internal.cnn.maskrcnn.smoothL1(YRPNRegDeltas, RPNRegressionTargets, RPNRegWeights);
        LossRPN = LossRPNClass + LossRPNReg;




        % Generate RCNN response targets
        %--------------------------------
        % Step 1: Match ground truth boxes to proposals 
        proposals1 = convertProposals(proposal1);
        proposals2 = convertProposals(proposal2);
        proposals3 = convertProposals(proposal3);
        proposals4 = convertProposals(proposal4);

        [assignment1, positiveIndex1, negativeIndex1] = vision.internal.cnn.maskrcnn.bboxMatchAndAssign(...
                                                                proposals1, gTruthBoxes,...
                                                                [0.5 1], [0 0.5],...
                                                                0);
        
        [assignment2, positiveIndex2, negativeIndex2] = vision.internal.cnn.maskrcnn.bboxMatchAndAssign(...
                                                                proposals2, gTruthBoxes,...
                                                                [0.6 1], [0 0.6],...
                                                                0);        
        
        [assignment3, positiveIndex3, negativeIndex3] = vision.internal.cnn.maskrcnn.bboxMatchAndAssign(...
                                                                proposals3, gTruthBoxes,...
                                                                [0.7 1], [0 0.7],...
                                                                0);    
        
        [assignment4, positiveIndex4, negativeIndex4] = vision.internal.cnn.maskrcnn.bboxMatchAndAssign(...
                                                                proposals4, gTruthBoxes,...
                                                                [0.7 1], [0 0.7],...
                                                                0);            

                                                                    
        % Step 2: Calcuate regression targets as (dx, dy, log(dw), log(dh))
        regressionTargets1 = vision.internal.cnn.maskrcnn.generateRegressionTargets(gTruthBoxes, proposals1,...
                                                                assignment1, positiveIndex1,...
                                                                obj.params.NumClasses);
        regressionTargets2 = vision.internal.cnn.maskrcnn.generateRegressionTargets(gTruthBoxes, proposals2,...
                                                                assignment2, positiveIndex2,...
                                                                obj.params.NumClasses);
        regressionTargets3 = vision.internal.cnn.maskrcnn.generateRegressionTargets(gTruthBoxes, proposals3,...
                                                                assignment3, positiveIndex3,...
                                                                obj.params.NumClasses);
        regressionTargets4 = vision.internal.cnn.maskrcnn.generateRegressionTargets(gTruthBoxes, proposals4,...
                                                                assignment4, positiveIndex4,...
                                                                obj.params.NumClasses);        
         
        classNames = categories(gTruthLabels{1});
        
        % Step 3: Assign groundtrutrh labels to proposals
        classificationTargets1 = vision.internal.cnn.maskrcnn.generateClassificationTargets (gTruthLabels, assignment1,...
                                                     positiveIndex1, negativeIndex1,...
                                                     classNames, obj.params.BackgroundClass);
        classificationTargets2 = vision.internal.cnn.maskrcnn.generateClassificationTargets (gTruthLabels, assignment2,...
                                                     positiveIndex2, negativeIndex2,...
                                                     classNames, obj.params.BackgroundClass);
        classificationTargets3 = vision.internal.cnn.maskrcnn.generateClassificationTargets (gTruthLabels, assignment3,...
                                                     positiveIndex3, negativeIndex3,...
                                                     classNames, obj.params.BackgroundClass);
        classificationTargets4 = vision.internal.cnn.maskrcnn.generateClassificationTargets (gTruthLabels, assignment4,...
                                                     positiveIndex4, negativeIndex4,...
                                                     classNames, obj.params.BackgroundClass);        
                                                 
        % Step 4: Calculate instance weights
        instanceWeightsReg1 = vision.internal.cnn.maskrcnn.regressionResponseInstanceWeights (classificationTargets1, obj.params.BackgroundClass);
        instanceWeightsReg2 = vision.internal.cnn.maskrcnn.regressionResponseInstanceWeights (classificationTargets2, obj.params.BackgroundClass);
        instanceWeightsReg3 = vision.internal.cnn.maskrcnn.regressionResponseInstanceWeights (classificationTargets3, obj.params.BackgroundClass);
        instanceWeightsReg4 = vision.internal.cnn.maskrcnn.regressionResponseInstanceWeights (classificationTargets4, obj.params.BackgroundClass);
         
        % Stage 2 (RCNN) Loss
        % --------------------       

        % *Classification loss*
        classificationTargets1 = cat(1, classificationTargets1{:})';
        classificationTargets2 = cat(1, classificationTargets2{:})';
        classificationTargets3 = cat(1, classificationTargets3{:})';
        classificationTargets4 = cat(1, classificationTargets4{:})';

        % onehotencode labels                       
        classificationTargets1 = onehotencode(classificationTargets1,1);
        classificationTargets2 = onehotencode(classificationTargets2,1);
        classificationTargets3 = onehotencode(classificationTargets3,1);
        classificationTargets4 = onehotencode(classificationTargets4,1);

        classificationTargets1(isnan(classificationTargets1)) = 0;
        classificationTargets2(isnan(classificationTargets2)) = 0;
        classificationTargets3(isnan(classificationTargets3)) = 0;
        classificationTargets4(isnan(classificationTargets4)) = 0;

        classificationTargets1 = reshape(classificationTargets1 ,1, 1, size(CCScore1,3),[]);
        classificationTargets2 = reshape(classificationTargets2 ,1, 1, size(CCScore2,3),[]);
        classificationTargets3 = reshape(classificationTargets3 ,1, 1, size(CCScore3,3),[]);
        classificationTargets4 = reshape(classificationTargets4 ,1, 1, size(YRCNNClass,3),[]);


        LossRCNNClass1 = vision.internal.cnn.maskrcnn.CrossEntropy(CCScore1, classificationTargets1);
        LossRCNNClass2 = vision.internal.cnn.maskrcnn.CrossEntropy(CCScore2, classificationTargets2);
        LossRCNNClass3 = vision.internal.cnn.maskrcnn.CrossEntropy(CCScore3, classificationTargets3);
        LossRCNNClass4 = vision.internal.cnn.maskrcnn.CrossEntropy(YRCNNClass, classificationTargets4);
         
        % *Weighted regression loss*
        regressionTargets1 = cat(1,regressionTargets1{:})';
        regressionTargets2 = cat(1,regressionTargets2{:})';
        regressionTargets3 = cat(1,regressionTargets3{:})';
        regressionTargets4 = cat(1,regressionTargets4{:})';

        regressionTargets1 = reshape(regressionTargets1, 1, 1, size(CCRegDeltas1,3),[]);
        regressionTargets2 = reshape(regressionTargets2, 1, 1, size(CCRegDeltas2,3),[]);
        regressionTargets3 = reshape(regressionTargets3, 1, 1, size(CCRegDeltas3,3),[]);
        regressionTargets4 = reshape(regressionTargets4, 1, 1, size(YRCNNReg,3),[]);
        


        instanceWeightsReg1 = cat(1, instanceWeightsReg1{:})';
        instanceWeightsReg2 = cat(1, instanceWeightsReg2{:})';
        instanceWeightsReg3 = cat(1, instanceWeightsReg3{:})';
        instanceWeightsReg4 = cat(1, instanceWeightsReg4{:})';

        instanceWeightsReg1 = reshape(instanceWeightsReg1, 1, 1, size(CCRegDeltas1,3),[]);
        instanceWeightsReg2 = reshape(instanceWeightsReg2, 1, 1, size(CCRegDeltas2,3),[]);
        instanceWeightsReg3 = reshape(instanceWeightsReg3, 1, 1, size(CCRegDeltas3,3),[]);
        instanceWeightsReg4 = reshape(instanceWeightsReg4, 1, 1, size(YRCNNReg,3),[]);
        

        LossRCNNReg1 = vision.internal.cnn.maskrcnn.smoothL1(CCRegDeltas1, single(regressionTargets1), single(instanceWeightsReg1));
        LossRCNNReg2 = vision.internal.cnn.maskrcnn.smoothL1(CCRegDeltas2, single(regressionTargets2), single(instanceWeightsReg2));
        LossRCNNReg3 = vision.internal.cnn.maskrcnn.smoothL1(CCRegDeltas3, single(regressionTargets3), single(instanceWeightsReg3));
        LossRCNNReg4 = vision.internal.cnn.maskrcnn.smoothL1(YRCNNReg, single(regressionTargets4), single(instanceWeightsReg4));

        %adjust weights for repeated iteration
        LossRCNNReg3 = LossRCNNReg3/2;
        LossRCNNReg4 = LossRCNNReg4/2;
        LossRCNNClass3 = LossRCNNClass3/2;
        LossRCNNClass4 = LossRCNNClass4/2;


        % Step 5: Generate mask targets
         
        % Crop and resize the instances based on proposal bboxes and network output size
        maskOutputSize = obj.params.MaskOutputSize;
        croppedMasks = vision.internal.cnn.maskrcnn.cropandResizeMasks (gTruthMasks, gTruthBoxes, maskOutputSize);
         
        % Generate mask targets
        maskTargets = vision.internal.cnn.maskrcnn.generateMaskTargets(croppedMasks, assignment1, classificationTargets1, obj.params);
        

        % Mask Loss (Weighted cross entropy)
        maskTargets= cat(4,maskTargets{:});
        positiveIndex1 = cat(1,positiveIndex1{:});
        LossRCNNMask = vision.internal.cnn.maskrcnn.SpatialCrossEntropy(YMask, single(maskTargets), positiveIndex1);
         
        




        % Total Stage 2 loss
        LossRCNN = LossRCNNReg1 + LossRCNNReg2 + LossRCNNReg3 + LossRCNNReg4 + LossRCNNClass1 + LossRCNNClass2 + LossRCNNClass3 + LossRCNNClass4 + LossRCNNMask;
        
         



        
        % Total Loss
        %------------
        loss = LossRCNN + LossRPN;
    
        lossData.Loss = loss;
        lossData.RPNClass = LossRPNClass;
        lossData.RPNReg = LossRPNReg;
        lossData.RPNLoss = LossRPN;
        lossData.Class1 = LossRCNNClass1;
        lossData.Class2 = LossRCNNClass2;
        lossData.Class3 = LossRCNNClass3;
        lossData.Class4 = LossRCNNClass4;
        lossData.Reg1 = LossRCNNReg1;
        lossData.Reg2 = LossRCNNReg2;
        lossData.Reg3 = LossRCNNReg3;
        lossData.Reg4 = LossRCNNReg4;
        lossData.MaskLoss = LossRCNNMask;

        end
    end

end

function proposals = convertProposals(proposal)
        % Proposals are 5XNumProposals (Due to batch restrictions from custom RPL layer)
        proposals = gather(extractdata(proposal));
        
        % Convert proposals to numProposals x 5 (as expected by the rest of post processing code)
        proposals =proposals';
        
        proposals(:,1:4) = vision.internal.cnn.maskrcnn.boxUtils.x1y1x2y2ToXYWH(proposals(:,1:4));
        
        numImagesInBatch = size(gTruthBoxes,1);
        %Convert numProposalsx5 Proposals to numImagesInBatchx1 (Group by image index)
        proposals = vision.internal.cnn.maskrcnn.groupProposalsByImageIndex(proposals, numImagesInBatch);
end