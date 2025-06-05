%trainMaskRCNN Train a Mask R-CNN network to perform instance segmentation.
%
% Use of this function requires that you have the Deep Learning Toolbox(TM).
%
% mrcnn = trainMaskRCNN(trainingData, network, options)
% trains a Mask R-CNN network object. A Mask R-CNN network object can be trained to
% perform instance segmentation to detect and segment multiple object classes.
%
% [..., info] = trainMaskRCNN(...) additionally returns
% information on training progress such as training loss and accuracy for
% each iteration. info is a struct array with each struct containing the
% following fields.
%  
%   LearnRate           - Learning rate at each iteration.
%   TrainingLoss        - Total Training loss at each iteration. This is the
%                         combination of the Region proposal network (RPN),
%                         classification, regression and mask segmentation 
%                         loss used to train the Mask R-CNN network.
%   TrainingRPNLoss     - Total Region proposal network loss at the end of
%                         each iteration.
%   TrainingRMSE        - Training RMSE (root mean square error) for the box
%                         regression layer.
%   TrainingMaskLoss    - Training cross-entropy loss for the mask segmentation
%                         branch.   
%   ValidationLoss      - Validation loss at each iteration.
%   ValidationRPNLoss   - Validation RPN loss at each iteration.
%   ValidationRMSE      - Validation RMSE at each iteration.
%   ValidationMaskLoss  - Validation loss for the mask segmentation branch.
%
% trainingData - This is a datastore with the following format.
%
%                Datastore format:
%                -----------------
%                A datastore that returns a cell array on the read
%                methods with four cells.
%                1st Cell  : A cell vector of images that can be grayscale,
%                            or a RGB image of size H-by-W-by-1 or H-by-W-by-3.
%                2nd Cell  : A M-by-4 matrix of [x, y, width, height]
%                            bounding boxes specifying object locations
%                            of M objects within each image.
%                3rd Cell  : A categorical vector of size M-by-1 containing the
%                            object class names. Note that all the categorical
%                            data returned by the datastore must have the same
%                            categories.
%                4th Cell  : A H-by-W-by-M logical array of M object masks.
%
%                An option is to use combine method to create a datastore to
%                return the above 4 columns of data on read:
%                     1. ds1, imageDatastore that can return the 1st column of data
%                     2. ds2, boxLabelDatastore that can return the 2nd and 3rd column of data.
%                     3. ds3, imageDatastore with a custom read function to return masks as 4th column.
%                     4. ds = combine(ds1, ds2, ds3);
%
% network        Specify the Mask R-CNN network to train. The network should be a
%                maskrcnn object.
%
% options        Training options defined by the trainingOptions function
%                from Deep Learning Toolbox. The training options define
%                the training parameters of the neural network.
%
% Transfer Learn & Fine Tune Mask R-CNN
% -------------------------------------
% [...] = trainMaskRCNN(trainingData, detector, options)
% continues training a Mask R-CNN detector. Use this syntax to transfer-learn
% on a pre-trained maskrcnn object or continue training a maskrcnn detector
% with additional training data or to perform more training iterations to improve
% detector accuracy.
%
% % Additional input arguments
% ----------------------------
% [...] = trainMaskRCNN(..., Name=Value) specifies
% additional name-value pair arguments described below:
%
% 'PositiveOverlapRange'     A two-element vector that specifies a range of
%                            bounding box overlap ratios between 0 and 1.
%                            Region proposals that overlap with ground
%                            truth bounding boxes within the specified
%                            range are used as positive training samples.
%                            Overlap ratio is computed using
%                            intersection-over-union between two bounding
%                            boxes.
%
%                            Default: [0.5 1]
%
% 'NegativeOverlapRange'     A two-element vector that specifies a range of
%                            bounding box overlap ratios between 0 and 1.
%                            Region proposals that overlap with ground
%                            truth bounding boxes within the specified
%                            range are used as negative training samples.
%                            Overlap ratio is computed using
%                            intersection-over-union between two bounding
%                            boxes.
%
%                            Default: [0.1 0.5]
%
% 'NumStrongestRegions'      The maximum number of strongest region
%                            proposals generated for each training image.
%                            Reduce this value to speed-up processing time
%                            at the cost of training accuracy. Set this to
%                            inf to use all region proposals.
%
%                            Default: 1000
%
% 'NumRegionsToSample'       The number of region proposals, K, to randomly
%                            sample from each training image. Reduce the
%                            value of K to reduce memory usage and speed-up
%                            training at the cost of training accuracy.
%
%                            Default: 128 
%
% 'FreezeSubNetwork'         A string or a cell-array of strings specifying
%                            the name of the sub-network/s, the weights of
%                            which will be frozen during training. The possible
%                            set of values are - "backbone", "rpn",
%                            ["backbone", "rpn"] and "none".
%
%                            - "backbone" corresponds to the feature extraction
%                                         layers, including the post-roialign feature
%                                         extraction layers.
%                            - "rpn"      corresponds to region proposal layers.
%                            - "none"     makes the entire network train-able.
%
%                            Default: "none"
%
% 'ExperimentMonitor'        maskrcnn training experiment monitoring,
%                            specified as an experiment. Monitor object for
%                            use with the Experiment Manager app. Use this
%                            object to track the progress of the training,
%                            update information fields in the results
%                            table, record values of the metrics used by
%                            the training, and produce training plots.
%                         
%                            The following information is monitored during
%                            training:
%                               - Training loss at each iteration.
%                               - Training accuracy at each iteration.
%                               - Training RMSE (root mean square error)
%                                 for the box regression layer.
%                               - Training loss from the mask segmentation
%                                 branch.
%                               - Learning rate at each iteration.
%                         
%                            When validation data is provided in the
%                            training options input, additional information
%                            is monitored:
%                         
%                               - Validation loss at each iteration.
%                               - Validation accuracy at each iteration.
%                               - Validation RMSE at each iteration.
%                               - Validation loss from the mask segmentation
%                                 branch.     
%                         
%                            Default: "none"
%
% 'AssignAllGTinRPN'        Whether to assign all anchor boxes as a
%                           positive example, regarless of IoU values
%
%                           Default: "True"


function [network, info] = trainMRCNN(trainingData, network, options, params)

    arguments
        trainingData {iValidateTrainingdata, mustBeScalarOrEmpty}
        network (1,1) MRCNN
        options (1,1) nnet.cnn.TrainingOptions
        params.PositiveOverlapRange  {iValidateOverlapRange,...
                                                            mustBeGreaterThanOrEqual(params.PositiveOverlapRange, 0),...
                                                            mustBeLessThanOrEqual(params.PositiveOverlapRange, 1)} = [0.5 1]
        params.NegativeOverlapRange  {iValidateOverlapRange,...
                                                            mustBeGreaterThanOrEqual(params.NegativeOverlapRange, 0),...
                                                            mustBeLessThanOrEqual(params.NegativeOverlapRange, 1),...
                                                            iValidateOverlapRange} = [0.1 0.5]
        params.NumStrongestRegions (1,1)  {mustBeNumeric, mustBePositive, mustBeNonsparse,...
                                           mustBeScalarOrEmpty, mustBeNonempty, ...
                                           mustBeNonNan, mustBeReal} = 1000
        params.NumRegionsToSample (1,1)        {mustBeNumeric, mustBePositive, mustBeNonsparse,...
                                                mustBeScalarOrEmpty, mustBeNonempty, ...
                                                mustBeNonNan, mustBeInteger, mustBeFinite, mustBeReal} = 128
        params.FreezeSubNetwork string {iValidateFreezeSubNetwork} = "none"
        params.ExperimentMonitor experiments.Monitor {mustBeScalarOrEmpty, iValidateMonitor} = experiments.Monitor.empty()
        params.ForcedPositiveProposals logical = 1;

    end

    % Validate no overlap between pos & neg overlap ranges
    iValidateNoRangeIntersection(params.PositiveOverlapRange, params.NegativeOverlapRange);

    % Validate if gpu execution on trainingOptions is supported
    if(strcmp(options.ExecutionEnvironment, 'gpu') && ~canUseGPU)
        error(message('vision:maskrcnn:noGPUFound'));
    end

    % Create minibatchqueue
    %preprocessDS = transform(trainingData, @(x)preprocessData(x, network.InputSize));

    if(~isempty(options.ValidationData))
        iValidateTrainingdata(options.ValidationData);
        % Add preprocessing to validation data
        %options.ValidationData = transform(options.ValidationData, @(x)preprocessData(x, network.InputSize));
    end

    miniBatchSize = options.MiniBatchSize;
    
    % Set output environment.
    if options.ExecutionEnvironment == "multi-gpu" 
        execEnvironment = "gpu";
    elseif options.ExecutionEnvironment == "parallel"
        execEnvironment = "auto";
    else
        execEnvironment = options.ExecutionEnvironment;
    end    
    miniBatchFcn = @(img,boxes,labels,masks) deal(cat(4,img{:}),boxes,labels,masks);

    mbqTrain = minibatchqueue(trainingData,4, ...
        "MiniBatchFormat",["SSCB","","",""], ...
        "MiniBatchSize",miniBatchSize, ...
        "OutputCast",["single","","",""], ...
        "OutputAsDlArray",[true,false,false,false], ...
        "MiniBatchFcn",miniBatchFcn, ...
        "OutputEnvironment",[execEnvironment,"cpu","cpu","cpu"]);
    
    % Create training configuration
    trainConfig = getMaskRCNNTrainConfig(network);
    trainConfig.PositiveOverlapRange = params.PositiveOverlapRange;
    trainConfig.NegativeOverlapRange = params.NegativeOverlapRange;
    trainConfig.NumRegionsToSample = double(params.NumRegionsToSample);
    trainConfig.ForcedPositiveProposals = params.ForcedPositiveProposals;

    % Configure maskrcnn object for training with network freeze options
    % and NumStrongestRegions
    network = configureForTraining(network,...
                                double(round(params.NumStrongestRegions)),...
                                params.FreezeSubNetwork);

    lossFcn = MRCNNLoss(trainConfig);

    % Define metric names. These are same as the lossObj field names as
    % returned by the lossfunction below. This avoids recompute oof the metrics.

    if strcmp(version('-release'),'2024b')
            metrics = {images.dltrain.internal.LossMetricAdapter(lossFcn)};
    elseif strcmp(version('-release'),'2023a')
            metrics = struct('Loss',lossFcn, 'RPNClass', [], 'RPNReg', [], 'Class', [], "Reg", [], "MaskLoss", []);
    end

    [network,info] = images.dltrain.internal.dltrain(mbqTrain,network,options,lossFcn,metrics,'Loss', 'ExperimentMonitor',params.ExperimentMonitor);

    % Remove fields from info struct
    fieldsToRemove = {'Epoch', 'Iteration', 'TimeElapsed', 'OutputNetworkIteration'};
    info = rmfield(info, fieldsToRemove);
end

function data = preprocessData(data,targetSize)
    % Resize the image and scale the corresponding bounding boxes.
    
    if iscell(data)
        % Preprocess inputs while training - One observation at a
        % time
        % Assumption here is that the training data has the
        % following order - image, boxes, labels, masks

        I = data{1};
        info.OriginalSize = size(I);

        % Handle grayscale inputs
        % if(size(I,3)==1)
        %     I = repmat(I,[1 1 4 1]);
        % end
                        
        % Resize Images
        if(~isempty(data{1}))
            [data{1}, scale] = resizePreservingAspectRatio(single(I),targetSize(1:2),0);
            info.Scale = scale;
        else
            info.Scale = 1;
        end
        
        % Resize Boxes
        data{2} = bboxresize(data{2}, 1/info.Scale);

        % Resize Masks
        if(~isempty(data{4}))
            data{4} = logical(resizePreservingAspectRatio(uint8(data{4}), targetSize(1:2),0));
        end
       
    end
    
    % Normalize image using COCO imageset means

    
    im = data{1};
    
    % Step 3: Normalize
    for i = size(im,3)
        im(:,:,i) = im(:,:,i) - mean(im(:,:,i));
    end
    
    
    data{1} = im;
end

function trainConfig = getMaskRCNNTrainConfig(network)
    % Network parameters
    trainConfig.ImageSize = network.InputSize;
    trainConfig.NumClasses = numel(network.ClassNames);
    trainConfig.ClassNames = {network.ClassNames, 'background'};
    trainConfig.BackgroundClass = 'background';
    trainConfig.ClassAgnosticMasks = false;  
    trainConfig.ScaleFactor = [1 1]/32;

    %TODO: remove and read from net
    trainConfig.MaskOutputSize = [14 14];
       
    % Region Proposal network params
    trainConfig.AnchorBoxes = network.AnchorBoxes;
    trainConfig.NumAnchors = size(network.AnchorBoxes,1);
    trainConfig.NumRegionsToSample = 200;

    % NMS threshold
    trainConfig.OverlapThreshold = 0.7;
    trainConfig.MinScore = 0;
    trainConfig.RPNClassNames = {'Foreground', 'Background'};
    trainConfig.RPNBoxStd   = [1 1 1 1];
    trainConfig.RPNBoxMean  = [0 0 0 0];

    % RPN training target generation params
    trainConfig.RandomSelector = vision.internal.rcnn.RandomSelector();
    trainConfig.StandardizeRegressionTargets = false;
    trainConfig.MiniBatchPadWithNegatives = true;
    trainConfig.ProposalsOutsideImage = 'clip';
    trainConfig.BBoxRegressionNormalization = 'valid';
    trainConfig.RPNROIPerImage = trainConfig.NumRegionsToSample;
    trainConfig.CategoricalLookup = reshape(categorical([1 2 3],[1 2],trainConfig.RPNClassNames),[],1);

end

function [resizedImage, scale] = resizePreservingAspectRatio(img, targetSize, pad)
    % Compute the scale to resize the groundtruth bounding boxes.

    img = single(img);
    imgSize = size(img);
    
    % Compute Aspect Ratio.
    imgAspectRatio = imgSize(2)/imgSize(1);

    resizeRowsToTargetOutputSize = ceil(imgAspectRatio*targetSize(1));
    resizeColsToTargetOutputSize = ceil(targetSize(2)/imgAspectRatio);
    padSizeIfResizeRowsToTarget = resizeRowsToTargetOutputSize-targetSize(2);
    padSizeIfResizeColsToTarget = resizeColsToTargetOutputSize-targetSize(1);

    % Resize and pad image to final size
    if padSizeIfResizeRowsToTarget < padSizeIfResizeColsToTarget
        scale = imgSize(1)/targetSize(1);
        resizedImage = imresize(img,[targetSize(1),nan]);
        resizedImage(:,end+1:targetSize(2),:) = pad;
    elseif padSizeIfResizeColsToTarget < padSizeIfResizeRowsToTarget
        scale = imgSize(2)/targetSize(2);
        resizedImage = imresize(img,[nan,targetSize(2)]);
        resizedImage(end+1:targetSize(1),:,:) = pad;
    else
        scale = imgSize(1)/targetSize(1);
        resizedImage = imresize(img,targetSize(1:2));
    end

end

function flag = iValidateTrainingdata(trainData)

    if(~matlab.io.datastore.internal.shim.isDatastore(trainData))
        error(message('vision:maskrcnn:invalidDatastore'));
    end

    data = preview(trainData);
    % data is a cell array of size 1x4
    validateattributes(data,{'cell'}, {'size', [1 4]}, mfilename, 'trainingData',1);

    % validate training images
    %validateattributes(data{1},{'numeric'}, {'size', [NaN NaN 3]});

    %validate boxes
    validateattributes(data{2},{'numeric'}, {'size', [NaN 4]});

    numObjects = size(data{2},1);

    % validate labels
    validateattributes(data{3},{'categorical'}, {'column', 'numel', numObjects});

    % validate masks
    validateattributes(data{4},{'logical'}, {'size', [NaN NaN numObjects]});
    
    flag = true;
end

function iValidateFreezeSubNetwork(freezeStr)
    if isscalar(freezeStr)
        validatestring(lower(freezeStr), ["backbone", "rpn", "none"], mfilename,"FreezeSubNetwork");
    else
        if(~all(sort(lower(freezeStr)) == ["backbone", "rpn"]))
            error(message('vision:maskrcnn:invalidFreezeNetStr'));
        end
    end
end

function iValidateOverlapRange(x)

validateattributes(x,{'numeric'},{'size',[1, 2], 'real', 'finite', 'increasing', 'nonnan', 'nonsparse'});

end

function iValidateNoRangeIntersection(posRange, negRange)
    if(any(negRange > posRange(1)))
        error(message('vision:maskrcnn:overlapRangeIntersection'));
    end
end

function iValidateMonitor(monitor)

    if(isempty(monitor))
        return;
    end
    if ~isempty(monitor.Metrics)
        error(message('vision:ObjectDetector:nonEmptyMetricsInfoInExperimentMonitor'));
    end
end
