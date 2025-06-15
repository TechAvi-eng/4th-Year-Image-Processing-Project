classdef CascadeRCNN < deep.internal.sdk.LearnableParameterContainer

    properties (SetAccess = public, GetAccess = public)
        % Mask R-CNN sub networks
        FeatureExtractionNet
        RegionProposalNet1
        RegionProposalNet2
        RegionProposalNet3
        PostPoolFeatureExtractionNet
        DetectionHeads
        MaskSegmentationHead
    end

    properties(Dependent, Hidden=false)
        AllLearnables
    end
    
    % Network properties
    properties(SetAccess = public)
        % ModelName       - string name for the detector object
        ModelName

        % ClassNames - Cell array of strings
        ClassNames

        % InputSize  - [H W C] network Input size.
        InputSize

        % AnchorBoxes- Mx2 anchor boxes for proposal generation.
        AnchorBoxes
    end

    properties (Access=public)
        %ImageMean for normalizing images. Default is the COCO dataset mean
        %ImageMean = single([123.675, 116.28, 103.53 100]);

        % ScaleFactor - Ratio of feature size to image Size
        ScaleFactor = [1 1]/16;
        
        % Number of classes on the detection  head
        NumClasses

    end

    % ROI Pooling properties
    properties(SetAccess=public, GetAccess = public)
        PoolSize = [14 14];
        MaskPoolSize = [14 14];
    end
    
    % Region proposal properties
    properties(Access=public)      
        ProposalsOutsideImage = 'clip'
        BoxFilterFcn =  @(a,b,c,d)fasterRCNNObjectDetector.filterBBoxesBySize(a,b,c,d)

        MinSize = [1 1]
        MaxSize = [inf inf]

        MinScore = 0;
        
        OverlapThresholdRPN = 0.7

        NumStrongestRegionsBeforeProposalNMS = 6000
        NumStrongestRegions = 1000

        RPNBoxStd = [1 1 1 1]
        RPNBoxMean= [0 0 0 0]
    end
    
    % Prediction params
    properties (SetAccess = public, GetAccess = public)
        BoxMean = [0 0 0 0];
        BoxStd = [1 1 1 1]
        ScoreThreshold
        NumStrongestRegionsPrediction = 1000
        OverlapThresholdPrediction = 0.5
        UseSelectStrongest
        MaskThreshold = 0.5
    end

    % properties needed by dltrain
    properties(SetAccess=public, Hidden=false)
        Learnables
        Layers
        State
        InputNames
        OutputNames
    end

    %Training params
    properties (Access=public)
        FreezeBackbone = false
        FreezeRPN = false
    end

    methods (Access=public)
        function [imageOut, ratio] = preprocessImage(~, imageIn, imageSize, ~)
            % preprocessImage- resize, normalize and pad the input image.
            
            % Step 1: Resize the image while maintaining the aspect ratio
            imSize = size(imageIn);
            maxSize = max(imSize(1), imSize(2));
            minSize = min(imSize(1), imSize(2));
            
            if maxSize/minSize <= imageSize(2)/imageSize(1)
                % dim 1 is dominant
                ratio = imageSize(1)/minSize;
            else
                % dim 2 is dominant
                ratio = imageSize(2)/maxSize;
            end
            
            imageOut = imresize(imageIn,[int32(ratio * imSize(1)) int32(ratio*imSize(2))],'nearest');
            imageOut = rescale(imageOut);
            
            % Convert grayscale images to 3 channel
            % if(size(imageIn,3)==1)
            %     imageIn = repmat(imageIn, [1 1 4]);
            % end
    
            % Step 2: Normalize (zero-center)
            % for i=size(imageIn, 3)
            %     imageIn(:,:,i)= imageIn(:,:,1) - mean( imageIn(:,:,1) );
            % end
    
    
            % Step 3: Make the input divisible by 32
            % sz = size(imageIn);
            % paddedH = int32(ceil(sz(1)/32)*32);
            % paddedW = int32(ceil(sz(2)/32)*32);
            % 
            % imageOut = zeros(paddedH,paddedW,size(imageIn, 3), 'like', imageIn);
            % imageOut(1:sz(1),1:sz(2),:) = imageIn;
            
            
        end
    end
    
    % Constructors
    methods
        function obj = CascadeRCNN(classNames, anchorBoxes, options)
            arguments
                classNames {validateClassNames, mustBeUniqueNames} = ''
                anchorBoxes (:,2){mustBeNumeric, mustBePositive, mustBeReal, mustBeFinite} = []
                options.InputSize {mustBeNumeric, mustBePositive, mustBeReal, mustBeFinite} = []
                options.PoolSize (1,2) {mustBeNumeric, mustBePositive, mustBeReal, mustBeFinite} = [14 14]
                options.MaskPoolSize (1,2) {mustBeNumeric, mustBePositive, mustBeReal, mustBeFinite} = [14 14]
                options.ModelName {mustBeTextScalar} = 'EfficientNet'
                options.ScaleFactor (1,2) {mustBeNumeric, mustBePositive, mustBeReal, mustBeFinite} = [1 1]/16
            end
            
            vision.internal.requiresNeuralToolbox(mfilename);
            
            % This is added to support load network workflows all the
            % weights and properties will be populated by the loadobj
            % method.

            % Load pre-trained network
            dir = which('SegmentCells'); %find and load network from path of all files
            dir = dir(1:end-14);
            data = load([dir+"/NetData/"+options.ModelName+"/dlnetPostFeature.mat"] );
            obj.PostPoolFeatureExtractionNet = data.dlnetPostFeature;
            data = load([dir+"/NetData/"+options.ModelName+"/dlnetFeature.mat"] );
            obj.FeatureExtractionNet = data.dlnetFeature;
            data = load([dir+"/NetData/"+options.ModelName+"/dlnetRPN.mat"] );

            obj.RegionProposalNet1 = data.dlnetRPN;
            
            data = load([dir+"/NetData/"+options.ModelName+"/dlnetRPNC.mat"] );
            obj.RegionProposalNet2 = data.dlnetRPNC;
            obj.RegionProposalNet3 = data.dlnetRPNC;

            data = load([dir+"/NetData/"+options.ModelName+"/dlnetDetectHead.mat"] );
            obj.DetectionHeads = data.dlnetDetectHead;
            data = load([dir+"/NetData/"+options.ModelName+"/dlnetMaskHead.mat"] );
            obj.MaskSegmentationHead = data.dlnetMaskHead;
            
            % Customize network for new input size
            if(isempty(options.InputSize))
                % Default input size = inputLayer size
                layer = obj.FeatureExtractionNet.Layers(1);
                obj.InputSize = layer.InputSize;
            else
                % Update the input layer with new size
                inputName = 'Input_data';
                conv1Name='conv1';
                obj.InputSize = options.InputSize;
                inputLayer = imageInputLayer(options.InputSize,'Normalization',"none",'Name', 'Input_data');
                conv1 = convolution2dLayer([7 7], 64,"Stride",[2 2], "DilationFactor", [1 1], NumChannels=options.InputSize(3),BiasInitializer="zeros",BiasLearnRateFactor=0);
                featureGraph = layerGraph(obj.FeatureExtractionNet);

                featureGraph = replaceLayer(featureGraph, inputName, inputLayer);
                featureGraph = replaceLayer(featureGraph, conv1Name, conv1);

                obj.FeatureExtractionNet = dlnetwork(featureGraph);
            end
            
            % Customize network for new classes
            if(isempty(classNames))
                obj.ClassNames = iGetCOCOClasses();
                numClasses = length(classNames);
                obj.NumClasses = numClasses;
            else
                if(iscategorical(classNames))
                    classNames = string(classNames);
                end

                if(isstring(classNames))
                    classNames = cellstr(classNames);
                end

                obj.ClassNames = classNames;
                numClasses = length(classNames);
                obj.NumClasses = numClasses;

                % update the detect branch architecture - regression head &
                % classification head & Mask head
                % Update regression conv layer              
                detGraph = layerGraph(obj.DetectionHeads);
                regConvNode = 'detectorRegOut';
                convReg = convolution2dLayer([1 1], numClasses*4, 'Name', 'detectorRegOut', 'WeightsInitializer','he');
                
                detGraph = replaceLayer(detGraph, regConvNode, convReg);
                    
                classConvNode = 'node_133';
                convClass = convolution2dLayer([1 1], numClasses+1, 'Name', 'node_133', 'WeightsInitializer','he',BiasInitializer="zeros",BiasLearnRateFactor=0);
                detGraph = replaceLayer(detGraph, classConvNode, convClass);

                obj.DetectionHeads = dlnetwork(detGraph, 'Initialize', false);

                maskGraph = layerGraph(obj.MaskSegmentationHead);
                convMaskNode = 'node_167';
                convMask = convolution2dLayer([1 1], numClasses, 'Name','node_167','WeightsInitializer','he',BiasInitializer="zeros",BiasLearnRateFactor=0);
                maskGraph = replaceLayer(maskGraph, convMaskNode, convMask);
                obj.MaskSegmentationHead = dlnetwork(maskGraph, 'Initialize', false);
            end
            
            % Customize network for anchor boxes
            if(isempty(anchorBoxes))
                obj.AnchorBoxes = iGetDefaultCOCOAnchorBoxes();
            else
                obj.AnchorBoxes = double(anchorBoxes);
                numAnchors = size(anchorBoxes,1);

                % update the RPN branch architecture
                rpnGraph1 = layerGraph(obj.RegionProposalNet1);
                rpnRegConvNode = 'RPNRegOut';
                convReg = convolution2dLayer([1 1], numAnchors*4, 'Name', 'RPNRegOut','WeightsInitializer','narrow-normal');
                rpnGraph1 = replaceLayer(rpnGraph1, rpnRegConvNode, convReg);      

                rpnClassConvNode = 'RPNClassOut';
                convClass = convolution2dLayer([1 1], numAnchors, 'Name', 'RPNClassOut','WeightsInitializer','narrow-normal',BiasInitializer="zeros",BiasLearnRateFactor=0);
                rpnGraph1 = replaceLayer(rpnGraph1, rpnClassConvNode, convClass);


                rpnGraphC = layerGraph(obj.RegionProposalNet2);

                % Store copies in cell array for easier processing
                rpnGraphs = {rpnGraph1, rpnGraphC, rpnGraphC};
                layerNames = {rpnGraphC.Layers.Name};
                
                for graphIdx = 2:3
                    currentGraph = rpnGraphs{graphIdx};
                    
                    for layerIdx = 1:length(layerNames)
                        newLayer = currentGraph.Layers(layerIdx);
                        newLayer.Name = [layerNames{layerIdx} '_' num2str(graphIdx)];
                        currentGraph = replaceLayer(currentGraph, layerNames{layerIdx}, newLayer);
                    end
                    
                    rpnGraphs{graphIdx} = currentGraph;
                end
                
                % Extract back to individual variables if needed
                %rpnGraph1 = rpnGraphs{1};
                rpnGraph2 = rpnGraphs{2};
                rpnGraph3 = rpnGraphs{3};

                obj.RegionProposalNet1 = dlnetwork(rpnGraph1, 'Initialize', false);
                obj.RegionProposalNet2 = dlnetwork(rpnGraph2, 'Initialize', false);
                obj.RegionProposalNet3 = dlnetwork(rpnGraph3, 'Initialize', false);

            end
    
            %correct scale factor for rounding when using a shape not
            %divisible by the scale factor
            %obj.ScaleFactor = ceil(obj.InputSize(1:2).*options.ScaleFactor)./obj.InputSize(1:2);

            obj.PoolSize = options.PoolSize;
            obj.MaskPoolSize = options.MaskPoolSize;
            obj.ModelName = options.ModelName;
            
            % Initialize the dlnetworks
            obj = initialize(obj);
            
        end
    end
    
    methods(Access=?CascadeRCNN)

        % Declare regionproposal method (definition in separate file)
        proposals = regionProposal(obj, regressionBatch, scoresBatch)

        % Declare roiAlignPooling method (definition in separate file)
        outFeatures = roiAlignPooling (obj, X, boxes, poolSize)
    end

    methods(Access=public)
        % initialize all sub - dlnetworks
        function obj = initialize(obj)
            
            dlX = dlarray(rand(obj.InputSize, 'single')*255, 'SSCB');

            dlFeatures = predict(obj.FeatureExtractionNet, dlX);
            
            % predict on RPN
            if(~obj.RegionProposalNet1.Initialized)
                obj.RegionProposalNet1 = initialize(obj.RegionProposalNet1, dlFeatures);
                obj.RegionProposalNet2 = initialize(obj.RegionProposalNet2, dlFeatures);
                obj.RegionProposalNet3 = initialize(obj.RegionProposalNet3, dlFeatures);

            end

            [dlRPNScores, dlRPNReg] = predict(obj.RegionProposalNet1, dlFeatures, 'Outputs',{'RPNClassOut', 'RPNRegOut'});
            

            % Call region proposal
            dlProposals = regionProposal(obj, dlRPNReg, dlRPNScores);
            
            dlPooled1 = roiAlignPooling(obj, dlFeatures, dlProposals, obj.PoolSize);


            [dlRPNScores2, dlRPNReg2] = predict(obj.RegionProposalNet2, dlPooled1, 'Outputs',{'RPNClassOut_2', 'RPNRegOut_2'});

            refinedProposals = applyRegressionToProposals(dlProposals, dlRPNReg2);




    

            
            dlFinalFeatures = predict(obj.PostPoolFeatureExtractionNet, dlPooled, dlPooled);
    
            % Box regression
            if(~obj.DetectionHeads.Initialized)
                obj.DetectionHeads = initialize(obj.DetectionHeads, dlFinalFeatures);
            end
    
            % Predict on mask branch to get h x w x numClasses x numProposals
            % cropped masks.
            if(~obj.MaskSegmentationHead.Initialized)
                obj.MaskSegmentationHead = initialize(obj.MaskSegmentationHead,...
                                                          dlFinalFeatures);
            end
            
        end
        
        function outputFeatures = predict(obj, dlX)
        
            dlFeatures = predict(obj.FeatureExtractionNet, dlX, 'Acceleration','auto');
    
            [dlRPNScores, dlRPNReg] = predict(obj.RegionProposalNet, dlFeatures, 'Outputs',{'RPNClassOut', 'RPNRegOut'});
            
            % Call region proposal
            dlProposals = regionProposal(obj, dlRPNReg, dlRPNScores);
            
            dlPooled = roiAlignPooling(obj, dlFeatures, dlProposals, obj.PoolSize);
            
            dlFinalFeatures = predict(obj.PostPoolFeatureExtractionNet, dlPooled, dlPooled, 'Acceleration','auto');
            
            [dlBoxReg, dlBoxScores] = predict(obj.DetectionHeads, dlFinalFeatures, 'Acceleration','auto', 'Outputs',{'detectorRegOut', 'detectorClassOut'});
            
            outputFeatures{1} = dlProposals;
            outputFeatures{2}= dlBoxReg;
            outputFeatures{3}= dlBoxScores;
            outputFeatures{4}= dlFeatures;

        end
        function outputFeatures = reusePredict(obj, dlX, knownBBoxes, numAdditionalProposals)
        
            dlFeatures = predict(obj.FeatureExtractionNet, dlX, 'Acceleration','auto');
            
            [dlRPNScores, dlRPNReg] = predict(obj.RegionProposalNet, dlFeatures, 'Outputs',{'RPNClassOut', 'RPNRegOut'});
            
            dlProposals = sequentialRegionProposal(obj, dlRPNReg, dlRPNScores, knownBBoxes, numAdditionalProposals); 
            
            dlPooled = roiAlignPooling(obj, dlFeatures, dlProposals, obj.PoolSize);
            
            dlFinalFeatures = predict(obj.PostPoolFeatureExtractionNet, dlPooled, dlPooled, 'Acceleration','auto');
            
            [dlBoxReg, dlBoxScores] = predict(obj.DetectionHeads, dlFinalFeatures, 'Acceleration','auto', 'Outputs',{'detectorRegOut', 'detectorClassOut'});
            
            outputFeatures{1} = dlProposals;
            outputFeatures{2}= dlBoxReg;
            outputFeatures{3}= dlBoxScores;
            outputFeatures{4}= dlFeatures;

        end


        
    end

    methods(Access=public)

        function [dlRPNScores,dlRPNReg,dlProposals,dlBoxScores,dlBoxReg,dlMasks,state] = forward(obj, dlX ,varargin)
        
            [dlX, localState] = forward(obj.FeatureExtractionNet, dlX, 'Acceleration','none');
            state = localState;
    
            % forward on RPN
            [dlRPNScores, dlRPNReg, localState] = forward(obj.RegionProposalNet, dlX, 'Acceleration','none','Outputs',{'RPNClassOut', 'RPNRegOut'});
            state = vertcat(state, localState);
            
            % Call region proposal
            dlProposals = regionProposal(obj, dlRPNReg, dlRPNScores);
            
            % Perform region pooling
            dlPooled = roiAlignPooling(obj, dlX, dlProposals, obj.PoolSize);
            
            % forward on post region pooling feature extractor
            [dlFinalFeatures, localState] = forward(obj.PostPoolFeatureExtractionNet, dlPooled, dlPooled, 'Acceleration','none');
            state = vertcat(state, localState);
            
            % forward on detection and segmentation heads
            [dlBoxReg, dlBoxScores] = predict(obj.DetectionHeads, dlFinalFeatures, 'Acceleration','auto', 'Outputs',{'detectorRegOut', 'detectorClassOut'});
            state = vertcat(state, localState);
            
            [dlMasks, localState] = forward(obj.MaskSegmentationHead, dlFinalFeatures, 'Acceleration','none');
            state = vertcat(state, localState); 

        end

        function varargout = segmentObjects(obj, im, options)
        
        arguments
            obj 
            im {}
            options.Threshold (1,1){mustBeNumeric, mustBePositive, mustBeLessThanOrEqual(options.Threshold, 1), mustBeReal} = 0.5
            options.NumStrongestRegions (1,1) {mustBeNumeric, mustBePositive, mustBeReal} = 1000
            options.SelectStrongest (1,1) logical = true
            options.MinSize (1,2) {mustBeNumeric, mustBePositive, mustBeReal, mustBeInteger} = [1,1]
            options.MaxSize (1,2) {mustBeNumeric, mustBePositive, mustBeReal, mustBeInteger} = obj.InputSize(1:2)
            options.ExecutionEnvironment {mustBeMember(options.ExecutionEnvironment,{'gpu','cpu','auto'})} = 'auto'
            options.WriteLocation {mustBeTextScalar} = fullfile(pwd,'SegmentObjectResults')
            options.MiniBatchSize (1,1) {mustBeNumeric, mustBePositive, mustBeReal, mustBeInteger} = 1
            options.NamePrefix {mustBeTextScalar} = "segmentObj"
            options.Verbose (1,1) {validateLogicalFlag} = true

        end    
        
        if(isequal(options.ExecutionEnvironment, 'auto'))
            if(canUseGPU)
                options.ExecutionEnvironment = 'gpu';
            end
        end
        
        % If writeLocation is set with a non-ds input, throw a warning
        if(~matlab.io.datastore.internal.shim.isDatastore(im) &&...
                ~strcmp(options.WriteLocation, fullfile(pwd,'SegmentObjectResults')))

            warning(message('vision:maskrcnn:WriteLocNotSupported'));
        end

            

        % Update the prediction parameters
        obj.ScoreThreshold = options.Threshold;
        obj.NumStrongestRegions = options.NumStrongestRegions;
        obj.UseSelectStrongest = options.SelectStrongest;
        obj.MinSize = options.MinSize;
        obj.MaxSize = options.MaxSize;
        
        masks = [];
        boxLabel = [];
        boxScore = [];
        boxes = [];

        % Check if the input image is a single image or a batch
        if(matlab.io.datastore.internal.shim.isDatastore(im))
            nargoutchk(0,1);
            [varargout{1:nargout}] = ...
                                        segmentObjectsInDatastore(obj, im,...
                                                                options.MiniBatchSize,...
                                                                options.NamePrefix,...
                                                                options.WriteLocation,...
                                                                options.Verbose,...
                                                                options.ExecutionEnvironment);
        elseif(ndims(im)<=3)
            nargoutchk(0,4);
            [varargout{1:nargout}] =...
                                         segmentObjectsInSingleImg(obj, im,...
                                                   options.ExecutionEnvironment);
        elseif(ndims(im)==4)
            nargoutchk(0,4);
            [varargout{1:nargout}] = ...
                                        segmentObjectsInImgStack(obj, im,...
                                                    options.MiniBatchSize,...
                                                    options.ExecutionEnvironment);  
        else
            % Code flow shouldn't reach here (ensured by validation code).
            assert(false, 'Invalid image input.');
        end
            
        end
        
  
        function varargout = segmentFrameviaKnowns(obj, frame, knownbboxes, options)
        
        arguments
            obj 
            frame {}
            knownbboxes (:,4) = []
            options.Threshold (1,1){mustBeNumeric, mustBePositive, mustBeLessThanOrEqual(options.Threshold, 1), mustBeReal} = 0.7
            options.NumStrongestRegions (1,1) {mustBeNumeric, mustBePositive, mustBeReal} = 1000
            options.NumAdditionalProposals (1,1) {mustBeNumeric, mustBePositive, mustBeReal} = max(size(knownbboxes, 1)*2, 100); %2/3 of proposals are new by (with a minimum of 100) by default
            options.SelectStrongest (1,1) logical = true
            options.MinSize (1,2) {mustBeNumeric, mustBePositive, mustBeReal, mustBeInteger} = [1,1]
            options.MaxSize (1,2) {mustBeNumeric, mustBePositive, mustBeReal, mustBeInteger} = obj.InputSize(1:2)
            options.ExecutionEnvironment {mustBeMember(options.ExecutionEnvironment,{'gpu','cpu','auto'})} = 'auto'
            options.WriteLocation {mustBeTextScalar} = fullfile(pwd,'SegmentObjectResults')
            options.MiniBatchSize (1,1) {mustBeNumeric, mustBePositive, mustBeReal, mustBeInteger} = 1
            options.NamePrefix {mustBeTextScalar} = "segmentObj"
            options.Verbose (1,1) {validateLogicalFlag} = true

        end    
        
        if(isequal(options.ExecutionEnvironment, 'auto'))
            if(canUseGPU)
                options.ExecutionEnvironment = 'gpu';
            end
        end
        
        % If writeLocation is set with a non-ds input, throw a warning
        if(~matlab.io.datastore.internal.shim.isDatastore(frame) &&...
                ~strcmp(options.WriteLocation, fullfile(pwd,'SegmentObjectResults')))

            warning(message('vision:maskrcnn:WriteLocNotSupported'));
        end

            
        % Update the prediction parameters
        obj.ScoreThreshold = options.Threshold;
        obj.NumStrongestRegions = options.NumStrongestRegions;
        obj.UseSelectStrongest = options.SelectStrongest;
        obj.MinSize = options.MinSize;
        obj.MaxSize = options.MaxSize;
        
        masks = [];
        boxLabel = [];
        boxScore = [];
        boxes = [];

            nargoutchk(0,4);
            [varargout{1:nargout}] = ...
                                        segmentObjectsReuseProposals(obj, frame, knownbboxes, ...
                                                                options.NumAdditionalProposals,...
                                                                options.ExecutionEnvironment);
            
        end
        
        function varargout = segmentFrame(obj, frame, RPNScores, options)
        
        arguments
            obj 
            frame {}
            RPNScores = [];
            options.Threshold (1,1){mustBeNumeric, mustBePositive, mustBeLessThanOrEqual(options.Threshold, 1), mustBeReal} = 0.5
            options.NumStrongestRegions (1,1) {mustBeNumeric, mustBePositive, mustBeReal} = 1000
            options.SelectStrongest (1,1) logical = true
            options.MinSize (1,2) {mustBeNumeric, mustBePositive, mustBeReal, mustBeInteger} = [1,1]
            options.MaxSize (1,2) {mustBeNumeric, mustBePositive, mustBeReal, mustBeInteger} = obj.InputSize(1:2)
            options.ExecutionEnvironment {mustBeMember(options.ExecutionEnvironment,{'gpu','cpu','auto'})} = 'auto'
            options.WriteLocation {mustBeTextScalar} = fullfile(pwd,'SegmentObjectResults')
            options.MiniBatchSize (1,1) {mustBeNumeric, mustBePositive, mustBeReal, mustBeInteger} = 1
            options.NamePrefix {mustBeTextScalar} = "segmentObj"
            options.Verbose (1,1) {validateLogicalFlag} = true
            options.Alpha (1,1) {mustBeReal, mustBeGreaterThanOrEqual(options.Alpha, 0), mustBeLessThanOrEqual(options.Alpha, 1)} = 0.15;

        end    
        
        if(isequal(options.ExecutionEnvironment, 'auto'))
            if(canUseGPU)
                options.ExecutionEnvironment = 'gpu';
            end
        end
        
        % If writeLocation is set with a non-ds input, throw a warning
        if(~matlab.io.datastore.internal.shim.isDatastore(frame) &&...
                ~strcmp(options.WriteLocation, fullfile(pwd,'SegmentObjectResults')))

            warning(message('vision:maskrcnn:WriteLocNotSupported'));
        end

            
        % Update the prediction parameters
        obj.ScoreThreshold = options.Threshold;
        obj.NumStrongestRegions = options.NumStrongestRegions;
        obj.UseSelectStrongest = options.SelectStrongest;
        obj.MinSize = options.MinSize;
        obj.MaxSize = options.MaxSize;
        
        masks = [];
        boxLabel = [];
        boxScore = [];
        boxes = [];
        RPN = [];

            nargoutchk(0,5);
            [varargout{1:nargout}] = ...
                                        segmentObjectsinVideo(obj, frame, RPNScores, options.Alpha, ...
                                                                options.ExecutionEnvironment);
            
        end
        
    end



         



    
    %======================================================================
    methods(Access=public)
        function s = saveobj(this)
            s.Version                      = 2.0;
            s.ModelName                    = this.ModelName;
            s.FeatureExtractionNet         = this.FeatureExtractionNet;
            s.RegionProposalNet1            = this.RegionProposalNet1;
            s.RegionProposalNet2            = this.RegionProposalNet2;
            s.RegionProposalNet3            = this.RegionProposalNet3;
            s.PostPoolFeatureExtractionNet = this.PostPoolFeatureExtractionNet;
            s.DetectionHeads               = this.DetectionHeads;
            s.MaskSegmentationHead         = this.MaskSegmentationHead;
            s.ClassNames                   = this.ClassNames;
            s.AnchorBoxes                  = this.AnchorBoxes;
            s.InputSize                    = this.InputSize;
            s.PoolSize                     = this.PoolSize;
            s.MaskPoolSize                 = this.MaskPoolSize;
            s.FreezeBackbone               = this.FreezeBackbone;
            s.FreezeRPN                    = this.FreezeRPN;
        end

        function obj = configureForTraining(obj, tNumStrongestRegions, freezeSubnetStr)
            
            % Set params for training
            obj.NumStrongestRegions = tNumStrongestRegions;
            
            if(lower(freezeSubnetStr)=="rpn")
                obj.FreezeRPN = true;
                obj.FreezeBackbone = false;
            elseif(lower(freezeSubnetStr)=="backbone")
                obj.FreezeBackbone = true;
                obj.FreezeRPN = false;
            elseif(lower(freezeSubnetStr)=="none")
                obj.FreezeBackbone = false;
                obj.FreezeRPN = false;
            elseif(all(sort(lower(freezeSubnetStr))==["backbone", "rpn"]))
                obj.FreezeBackbone = true;
                obj.FreezeRPN = true;
            else
                assert(false,"Invalid freezeSubnetStr");
            end

        end
        
    end
    
    methods(Static, Access=public)
        function this = loadobj(s)
            try
                vision.internal.requiresNeuralToolbox(mfilename);

                this = CascadeRCNN("none");
                   
                this.FeatureExtractionNet         = s.FeatureExtractionNet;
                this.RegionProposalNet1           = s.RegionProposalNet1;
                this.RegionProposalNet2           = s.RegionProposalNet2;
                this.RegionProposalNet3           = s.RegionProposalNet3;
                this.PostPoolFeatureExtractionNet = s.PostPoolFeatureExtractionNet;
                this.DetectionHeads               = s.DetectionHeads;
                this.MaskSegmentationHead         = s.MaskSegmentationHead;
                this.ClassNames                   = s.ClassNames;
                this.AnchorBoxes                  = s.AnchorBoxes;
                this.InputSize                    = s.InputSize;
                this.PoolSize                     = s.PoolSize;
                this.MaskPoolSize                 = s.MaskPoolSize;
                this.ModelName                    = s.ModelName;
                if(s.Version > 1.0)
                    this.FreezeBackbone               = s.FreezeBackbone;
                    this.FreezeRPN                    = s.FreezeRPN;
                end
            catch ME
                rethrow(ME)
            end
        end
    end

    methods(Access=public)
        
        function [masks, boxLabel, boxScore, boxes] = segmentObjectsInSingleImg(obj, im, executionEnvironment)
        
            orgSize = size(im);
    
            % Resize and normalize image
            [im, scaleRatio] = preprocessImage(obj, single(im), obj.InputSize);

            if(isequal(executionEnvironment, 'gpu'))
                %GPU
                im = gpuArray(im);
            else
                %Host
                if(isgpuarray(im))
                    im = extractData(im);
                end
            end
            
            dlX = dlarray(im, 'SSCB');
            netOut = predict(obj, dlX);
            imageSize = size(im);
            [boxes, boxLabel, boxScore] = postProcessOutputs(obj,...
                                          extractdata(netOut{2}),...
                                          extractdata(netOut{3}),...
                                          extractdata(netOut{1}), imageSize, 1);
            
            boxes = boxes{1};
            boxLabel = boxLabel{1};
            boxScore = boxScore{1};

            % If predicted boxes are empty, return empty mask & empty cat for
            % labels
            if(isempty(boxes))
                boxLabel = categorical(boxLabel, 1:numel(obj.ClassNames), obj.ClassNames);
                masks = zeros(0,0,0, 'like', boxes);
                return;
            end
            
            maskBranchBoxes = [boxes  ones(size(boxes,1), 1)];
            
            maskBranchBoxes = vision.internal.cnn.boxUtils.xywhToX1Y1X2Y2(...
                                                                         maskBranchBoxes);
    
            dlBoxes = dlarray(maskBranchBoxes', 'SSCB');
            
            % Feature pooling for mask branch
            dlMaskPooled = roiAlignPooling(obj, netOut{4}, dlBoxes, obj.MaskPoolSize);
            
            dlPostFeatureMask = predict(obj.PostPoolFeatureExtractionNet, dlMaskPooled, dlMaskPooled, 'Acceleration','auto');
            
            % Predict on mask branch to get h x w x numClasses x numProposals
            % cropped masks.
            dlMasks = predict(obj.MaskSegmentationHead, dlPostFeatureMask);
            dlMasks = extractdata(dlMasks);
    
            % Extract the cropped mask corresponding to the boxLabel for
            % each proposal.
            finalCroppedMasks = zeros(size(dlMasks,1), size(dlMasks, 2),...
                                      size(boxes,1));
            
            for i = 1:size(boxes,1)
                finalCroppedMasks(:,:,i) = dlMasks(:,:, boxLabel(i), i);
            end
            
            % Final box detections in original image coordinates
            if(isgpuarray(boxes))
                boxes = gather(boxes);
            end
            boxes = bboxResizePixel(obj, floor(boxes), 1/scaleRatio);

            boxes = clipBoxes(obj, boxes, orgSize);
            
            % Filter out zero width height predictions
            invalidBoxes = (boxes(:,3)==0 | boxes(:,4)==0);
            boxes(invalidBoxes,:)=[];
            boxLabel(invalidBoxes) = [];
            boxScore(invalidBoxes) = [];

            % Generate full sized masks from cropped masks
            masks = generateFullSizedMasks(obj, finalCroppedMasks, boxes, orgSize);
            
            % Convert boxLabels to categorical
            boxLabel = categorical(boxLabel, 1:numel(obj.ClassNames), obj.ClassNames);
        end

        function [masks, boxLabel, boxScore, boxes] = segmentObjectsInBatch(obj, im, executionEnvironment)
        
            if(iscell(im))
                batchSize = numel(im);
            else
                batchSize = size(im,4);
            end
            
            for batchIdx = 1:batchSize
                % Resize and normalize image
                if(iscell(im))
                    orgSize{batchIdx} = size(im{batchIdx}); %#ok<AGROW> 
                    [imResized(:,:,:,batchIdx), scaleRatio(batchIdx)] = preprocessImage(obj, single(im{batchIdx}), obj.InputSize); %#ok<AGROW> 
                else
                    orgSize{batchIdx} = size(im); %#ok<AGROW> 
                    [imResized(:,:,:,batchIdx), scaleRatio(batchIdx)] = preprocessImage(obj, single(im(:,:,:,batchIdx)), obj.InputSize); %#ok<AGROW> 
                end
            end
            
            if(isequal(executionEnvironment, 'gpu'))
                %GPU
                imResized = gpuArray(imResized);
            else
                %Host
                if(isgpuarray(imResized))
                    imResized = extractData(imResized);
                end
            end
            dlX = dlarray(imResized, 'SSCB');
            netOut = predict(obj, dlX);

            imageSize = size(imResized);

            proposals = extractdata(netOut{1});
            proposalBatchIdx = proposals(5,:);

            [boxes, boxLabel, boxScore] = postProcessOutputs(obj,...
                                          extractdata(netOut{2}),... % Reg deltas
                                          extractdata(netOut{3}),... % Scores
                                                       proposals,... % Proposals
                                            imageSize, batchSize);
        
                
            % If predicted boxes are empty, return empty mask & empty cat for
            % labels
            if(isempty(boxes))
                boxLabel = categorical(boxLabel, 1:numel(obj.ClassNames), obj.ClassNames);
                masks = zeros(0,0,0, 'like', boxes);
                return;
            end
            
            % create a flattened list of boxes
            maskBranchBoxes = [];
            maskBranchLabels = [];
            for idx = 1:length(boxes)
                if(~isempty(boxes{idx}))
                    maskBranchBoxes = [maskBranchBoxes ; [boxes{idx}  idx*ones(size(boxes{idx},1), 1)]];
                    maskBranchLabels = [maskBranchLabels; boxLabel{idx}];
                end
            end

            % If boxes in the entire batch are empty, return empty masks
            % and empty categorical labels for the batch
            if(isempty(maskBranchBoxes))
                boxLabel = categorical(boxLabel{1}, 1:numel(obj.ClassNames), obj.ClassNames);
                boxLabel = repmat({boxLabel}, 1, length(boxes));
                masks = zeros(0,0,0, 'like', boxes{1});
                masks = repmat({masks}, 1, length(boxes));
                return;
            end
            
            maskBranchBoxes = vision.internal.cnn.boxUtils.xywhToX1Y1X2Y2(...
                                                                         maskBranchBoxes);
    
            dlBoxes = dlarray(maskBranchBoxes', 'SSCB');
            
            % Feature pooling for mask branch
            dlMaskPooled = roiAlignPooling(obj, netOut{4}, dlBoxes, obj.MaskPoolSize);
            
            dlPostFeatureMask = predict(obj.PostPoolFeatureExtractionNet, dlMaskPooled, dlMaskPooled, 'Acceleration','auto');
            
            % Predict on mask branch to get h x w x numClasses x numProposals
            % cropped masks.
            dlMasks = predict(obj.MaskSegmentationHead, dlPostFeatureMask);
            dlMasks = extractdata(dlMasks);
    
            % Extract the cropped mask corresponding to the boxLabel for
            % each proposal.
            finalCroppedMasks = zeros(size(dlMasks,1), size(dlMasks, 2),...
                                      size(maskBranchBoxes,1));
            
            for i = 1:size(maskBranchBoxes,1)
                finalCroppedMasks(:,:,i) = dlMasks(:,:, maskBranchLabels(i), i);
            end
            
            % group masks in a cell array by the image idx
            imIdx = maskBranchBoxes(:,5);

            for idx = 1:batchSize
                % Final box detections in original image coordinates
                if(isgpuarray(boxes{idx}))
                    boxes{idx} = gather(boxes{idx});
                end
                boxes{idx} = bboxResizePixel(obj, floor(boxes{idx}), 1/scaleRatio(idx));
    
                boxes{idx} = clipBoxes(obj, boxes{idx}, orgSize{idx});
                % Generate full sized masks from cropped masks
                masks{idx} = generateFullSizedMasks(obj, finalCroppedMasks(:,:,imIdx==idx), boxes{idx}, orgSize{idx});
                
                % Convert boxLabels to categorical
                boxLabel{idx} = categorical(boxLabel{idx}, 1:numel(obj.ClassNames), obj.ClassNames);
            end
        end

        function [masks, boxLabel, boxScores, boxes] = segmentObjectsInImgStack(obj, im, miniBatchSize, executionEnvironment)
            
            stackSize = size(im,4);

            masks = [];
            boxes = [];
            boxLabel = [];
            boxScores = [];

            % Process images from the imageStack
            for startIdx = 1 : miniBatchSize : stackSize
                
                endIdx = min(startIdx+miniBatchSize-1, stackSize);

                imBatch = im(:,:,:,startIdx:endIdx);
               
                [masksCell, boxLabelCell, boxScoreCell, boxesCell] = ...
                                            segmentObjectsInBatch(obj, imBatch, executionEnvironment); 

                masks = [masks; masksCell']; %#ok<AGROW> 
                boxes = [boxes; boxesCell']; %#ok<AGROW> 
                boxLabel = [boxLabel; boxLabelCell']; %#ok<AGROW> 
                boxScores = [boxScores; boxScoreCell']; %#ok<AGROW> 
            end
        end
    
        function outds = segmentObjectsInDatastore(obj, imds, miniBatchSize, namePrefix, writeLocation, verboseFlag, executionEnvironment)
            
            imdsCopy = copy(imds);
            imdsCopy.reset();

            % Get a new write location
            fileLocation = vision.internal.GetUniqueFolderName(writeLocation);
            
            if(~exist(fileLocation, 'dir'))
                success = mkdir(fileLocation);
                if(~success)
                    throwAsCaller(MException('vision:maskrcnn:folderCreationFailed',...
                           vision.getMessage('vision:maskrcnn:folderCreationFailed')));
                end
            end

            % Handle verbose display
            printer = vision.internal.MessagePrinter.configure(verboseFlag);
            
            printer.linebreak();
            iPrintHeader(printer);
            msg = iPrintInitProgress(printer,'', 1);
         
            imIdx = 0;
            outFileList = [];
            % Process images from the datastore
            while (hasdata(imdsCopy))
                
                imBatch = [];
                fileNames = []; % Needed to build output names for result .matfiles

                % Build a minibatch worth of data
                for i = 1:miniBatchSize
                    if(~hasdata(imdsCopy))
                        break;
                    end
                    imIdx = imIdx + 1;
                    [img, imgInfo] = read(imdsCopy); %#ok<AGROW> 
                    
                    % Handle combineDS - use first cell, as the image is
                    % expected to be the first output.
                    if(iscell(imgInfo))
                        imgInfo = imgInfo{1};
                    end
                    
                    %If the datastore doesn't expose filename, use the
                    % read index instead
                    if (isfield(imgInfo, 'Filename'))
                        [~,fileNames{i}] = fileparts(imgInfo.Filename); %#ok<AGROW> 
                    else
                        fileNames{i} = num2str(imIdx); %#ok<AGROW> 
                    end
                    
                    if(iscell(img))
                        imBatch{i} = img{1}; % image should be the first output
                    else
                        imBatch{i} = img;
                    end
                end
               
                [masksCell, boxLabelCell, boxScoreCell, boxesCell] = ...
                                            segmentObjectsInBatch(obj, imBatch, executionEnvironment); 

                % Write results to the disk
                for idx = 1:numel(masksCell)
                    
                    matfilename = string(namePrefix)+"_"+string(fileNames{idx})+".mat";    
                    
                    masks = masksCell{idx};
                    boxes = boxesCell{idx};
                    boxScore = boxScoreCell{idx};
                    boxLabel = boxLabelCell{idx};
                    
                    currFilename = fullfile(fileLocation, matfilename);

                    save(currFilename,...
                        "masks","boxes","boxScore","boxLabel");

                    outFileList = [outFileList; currFilename];
                end
                % Print number of processed images
                msg = iPrintProgress(printer, msg, imIdx+numel(masksCell)-1);
            end

            outds = fileDatastore(outFileList, 'FileExtensions', '.mat',...
                                                     'ReadFcn', @(x)vision.internal.SegmentObjectsReader(x));

            printer.linebreak(2);

        end
        
        function [masks, boxLabel, boxScore, boxes] = segmentObjectsReuseProposals(obj, im, knownbboxes, numAdditionalProposals, executionEnvironment)
        
            orgSize = size(im);
    
            % Resize and normalize image
            [im, scaleRatio] = preprocessImage(obj, single(im), obj.InputSize);

            if(isequal(executionEnvironment, 'gpu'))
                %GPU
                im = gpuArray(im);
            else
                %Host
                if(isgpuarray(im))
                    im = extractData(im);
                end
            end
            
            dlX = dlarray(im, 'SSCB');
            netOut = reusePredict(obj, dlX, knownbboxes, numAdditionalProposals);
            imageSize = size(im);
            [boxes, boxLabel, boxScore] = postProcessOutputs(obj,...
                                          extractdata(netOut{2}),...
                                          extractdata(netOut{3}),...
                                          extractdata(netOut{1}), imageSize, 1);
            
            boxes = boxes{1};
            boxLabel = boxLabel{1};
            boxScore = boxScore{1};

            % If predicted boxes are empty, return empty mask & empty cat for
            % labels
            if(isempty(boxes))
                boxLabel = categorical(boxLabel, 1:numel(obj.ClassNames), obj.ClassNames);
                masks = zeros(0,0,0, 'like', boxes);
                return;
            end
            
            maskBranchBoxes = [boxes  ones(size(boxes,1), 1)];
            
            maskBranchBoxes = vision.internal.cnn.boxUtils.xywhToX1Y1X2Y2(...
                                                                         maskBranchBoxes);
    
            dlBoxes = dlarray(maskBranchBoxes', 'SSCB');
            
            % Feature pooling for mask branch
            dlMaskPooled = roiAlignPooling(obj, netOut{4}, dlBoxes, obj.MaskPoolSize);
            
            dlPostFeatureMask = predict(obj.PostPoolFeatureExtractionNet, dlMaskPooled, dlMaskPooled, 'Acceleration','auto');
            
            % Predict on mask branch to get h x w x numClasses x numProposals
            % cropped masks.
            dlMasks = predict(obj.MaskSegmentationHead, dlPostFeatureMask);
            dlMasks = extractdata(dlMasks);
    
            % Extract the cropped mask corresponding to the boxLabel for
            % each proposal.
            finalCroppedMasks = zeros(size(dlMasks,1), size(dlMasks, 2),...
                                      size(boxes,1));
            
            for i = 1:size(boxes,1)
                finalCroppedMasks(:,:,i) = dlMasks(:,:, boxLabel(i), i);
            end
            
            % Final box detections in original image coordinates
            if(isgpuarray(boxes))
                boxes = gather(boxes);
            end
            boxes = bboxResizePixel(obj, floor(boxes), 1/scaleRatio);

            boxes = clipBoxes(obj, boxes, orgSize);
            
            % Filter out zero width height predictions
            invalidBoxes = (boxes(:,3)==0 | boxes(:,4)==0);
            boxes(invalidBoxes,:)=[];
            boxLabel(invalidBoxes) = [];
            boxScore(invalidBoxes) = [];

            % Generate full sized masks from cropped masks
            masks = generateFullSizedMasks(obj, finalCroppedMasks, boxes, orgSize);
            
            % Convert boxLabels to categorical
            boxLabel = categorical(boxLabel, 1:numel(obj.ClassNames), obj.ClassNames);
        end




        
        function [finalBoxes, finalLabels, finalScores] = postProcessOutputs(obj, deltas, scores, proposals, imageSize, batchSize)
            % postProcessOutputs converts network outputs to box detections
            % deltas - numClasses*4 x numProposals
            % scores - numClasses x numProposals
            % proposals - 5 x numProposals
    
            % deltas hold the box regressions in the format [dx1 dy1 dw1 dh1 ...
            % ... dx_numProposals dy_numProposals dw_numProposals dh_numProposals]
            
            deltas = squeeze(deltas);
            scores = squeeze(scores);
            proposalImageIdx = proposals(5,:);
            proposals = proposals';
            
            % Network outputs proposals in [x1 y1 x2 y2] format in the image
            % space. Convert to [x y w h] for further processing.
            proposals = vision.internal.cnn.boxUtils.x1y1x2y2ToXYWH(proposals);
            
            numClasses = length(obj.ClassNames);
    
            % Step 1: apply regression weights
            deltas(1:4:numClasses*4,:) = deltas(1:4:numClasses*4,:) * obj.BoxStd(1) + obj.BoxMean(1);
            deltas(2:4:numClasses*4,:) = deltas(2:4:numClasses*4,:) * obj.BoxStd(2) + obj.BoxMean(2);
            deltas(3:4:numClasses*4,:) = deltas(3:4:numClasses*4,:) * obj.BoxStd(3) + obj.BoxMean(3);
            deltas(4:4:numClasses*4,:) = deltas(4:4:numClasses*4,:) * obj.BoxStd(4) + obj.BoxMean(4);
            
            % Step 2: Apply regression to proposals
            boxes = applyRegression(obj, proposals, deltas);%, params.MinSize, params.MaxSize);
    
            % Step 3: convert boxes from numClasses*4 x numProposals to
            % 4xnumClassesxnumProposals
            boxes = reshape(boxes, 4, [], size(proposals,1));
            
            % Step 4: Remove scores corresponding to background & get the label
            % with max score
            backgroundIdx = numClasses+1;
            scores(backgroundIdx, :) = [];
            [boxScore, boxLabel] = max(scores, [], 1);
    
            % Step 5: Keep boxes corresponding to the highest score
            boxesTrim = zeros(4, size(boxes,3), 'like', boxes);
            for i = 1:size(boxes,3)
                boxesTrim(:,i) = squeeze(boxes(:,boxLabel(i), i));
            end
            boxes = boxesTrim;
            
            % Step 6: Filter out boxes with lowscores & outside the image
            lowScores = boxScore <= obj.ScoreThreshold;
            invalidBoxes = getInvalidBoxesAndScores(obj, boxes, boxScore, imageSize);
            
            removeIdx = lowScores|invalidBoxes;
            boxes(:, removeIdx) = [];
            boxScore(removeIdx) = [];
            boxLabel(removeIdx) = [];
            proposalImageIdx(removeIdx) = [];
    
            % Step 7: Clip the boxes to image bounds
            % the boxes here get flipped from 4xnumPropoals to numProposalsx4
            boxes = clipBoxes(obj, boxes', imageSize);

            % Step 8: Group predictions by image and apply NMS

            for imageIdx = 1:batchSize
                
                boxIdxs = (proposalImageIdx==imageIdx);
                
                finalBoxes{imageIdx} = boxes(boxIdxs,:);
                finalScores{imageIdx} = boxScore(boxIdxs);
                finalLabels{imageIdx} = boxLabel(boxIdxs);
    
                % Step 9: NMS per image
                if(obj.UseSelectStrongest)
                    [finalBoxes{imageIdx},finalScores{imageIdx},finalLabels{imageIdx}] = ...
                    selectStrongestBboxMulticlass(gather(finalBoxes{imageIdx}),...
                                                    gather(finalScores{imageIdx}'),...
                                                    gather(finalLabels{imageIdx}'),...
                                                    'RatioType', 'Union', ...
                                                    'OverlapThreshold', obj.OverlapThresholdPrediction,...
                                                    'NumStrongest', obj.NumStrongestRegionsPrediction);
                end
                
            end

        end
        
        function bboxB = bboxResizePixel(~,bboxA,scale)
            % bboxResizePixel resizes the bbox in pixel coordinates space.
            % This is a stand-in implementation for resize until we move to
            % spatial coordinates (when we will use the bboxresize function)

            % Calculate foward mapping such that 0.5 in input space maps to 0.5 in
            % output space, and 1.5 in input space maps to 0.5+scale in output space.
            % This is the mapping used by IMRESIZE.
            sy = scale;
            sx = scale;
            
            tx = (1-sx)*0.5;
            ty = (1-sy)*0.5;
            
            % Convert xywh to box corner format. Place box corners at pixel centers in
            % spatial coordinates.
            boxes = bboxA(:,[1 2]);
            boxes = [boxes boxes + bboxA(:,[3 4]) - 1];
            
            % Scale boxes
            boxes = boxes .* [sx sy sx sy] + [tx ty tx ty];
            bboxB = max(1,round(boxes));
            
            % Covert from box corner to xywh.
            bboxB(:,[3 4]) = bboxB(:,[3 4]) - bboxB(:,[1 2]) + 1;
            
        end
 

        function boxes = applyRegression(obj, proposals, deltas)
        % applyRegression applies regression deltas to the input proposals- boxes
        % deltas - numClasses*4 x numProposals (regression for each box, for each class)
        % proposals - numProposals x 5
        % boxes - numClasses*4 x numProposals

        boxes = zeros(size(deltas), 'like', deltas);

        dx = deltas(1:4:end,:);
        dy = deltas(2:4:end,:);
        dw = deltas(3:4:end,:);
        dh = deltas(4:4:end,:);

        % Center of proposals
        cx = proposals(:,1) + proposals(:,3)*0.5;
        cy = proposals(:,2) + proposals(:,4)*0.5; 
        
        % Apply regression
        
        gx = dx.*proposals(:,3)' + cx';
        gy = dy.*proposals(:,4)' + cy';
        gw = proposals(:,3)' .* exp(dw);
        gh = proposals(:,4)' .* exp(dh);
        
        boxes(1:4:end, :) = gx - 0.5*gw;
        boxes(2:4:end, :) = gy - 0.5*gh;
        boxes(3:4:end, :) = gw;
        boxes(4:4:end, :) = gh;

        end

        function invalidIdx = getInvalidBoxesAndScores(~, boxes, scores, imageSize)
        % getInvalidBoxesAndScores, returns indices of boxes completely outside the
        % image or with invalid scores (infs and NaNs)
        x1 = boxes(1,:);
        y1 = boxes(2,:);
        x2 = boxes(3,:) + x1 - 1;
        y2 = boxes(4,:) + y1 - 1;
    
        boxOverlapsImage = ...
            (x1 < imageSize(2) & x2 > 1) & ...
            (y1 < imageSize(1) & y2 > 1);

        invalidIdx = ~boxOverlapsImage;

        invalidIdx = invalidIdx | ~all(isfinite(boxes),1);
        invalidIdx = invalidIdx | ~(isfinite(scores));

        end

        function bbox = clipBoxes(obj, bbox, imgSize)
        % clipBoxes, clips the boxes to lower and upper image bounds.
        % bbox - Nx4 box coordinates ordered as  [x y w h].
        % Assumption: TThe original bounding boxes all overlap the image. 
    
        % Get coordinates of upper-left (x1,y1) and bottom-right (x2,y2) corners. 
        x1 = bbox(:,1);
        y1 = bbox(:,2);
    
        offset = 1;
    
        x2 = bbox(:,1) + bbox(:,3) - offset;
        y2 = bbox(:,2) + bbox(:,4) - offset;
    
        x1(x1 < 1) = 1;
        y1(y1 < 1) = 1;

        x1(x1>imgSize(2)) = imgSize(2);
        y1(y1>imgSize(1)) = imgSize(1);


        x2(x2 < 1) = 1;
        y2(y2 < 1) = 1;

        x2(x2 > imgSize(2)) = imgSize(2);
        y2(y2 > imgSize(1)) = imgSize(1);
    
            
        bbox = [x1 y1 x2-x1+offset y2-y1+offset];
        end

        function masks = generateFullSizedMasks(obj, croppedMasks, boxes, imageSize)
        % generateFullSizedMasks inserts croppedMaks back into a full sized
        % mask using the object locations specified in boxes
        % croppedMasks - h x w x numObjects
        % boxes        - numObjects x 4
        % masks        - imageSize(1) x imageSize(2) x numObjects
        
        numObjects = size(boxes,1);
        masks = zeros([imageSize(1:2) numObjects], 'logical');
    
        for i=1:numObjects
    
            m = imresize(croppedMasks(:,:,i), [boxes(i,4) boxes(i,3)],'cubic') > obj.MaskThreshold ;
            masks(boxes(i,2):boxes(i,2)+boxes(i,4)-1, ...
                boxes(i,1):boxes(i,1)+boxes(i,3)-1, i) = m;
    
        end

        end

    end
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Learnables update methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        function s = get.AllLearnables(obj)
            % Package all subnetwork learnables into a structure
            if(~obj.FreezeBackbone)
                s.FeatureExtractionNet = obj.FeatureExtractionNet.Learnables;
                s.PostPoolFeatureExtractionNet = obj.PostPoolFeatureExtractionNet.Learnables;
            end

            if(~obj.FreezeRPN)
                s.RegionProposalNet1 = obj.RegionProposalNet1.Learnables;
                s.RegionProposalNet2 = obj.RegionProposalNet2.Learnables;
                s.RegionProposalNet3 = obj.RegionProposalNet3.Learnables;
            end
            
            s.DetectionHeads = obj.DetectionHeads.Learnables;
            s.MaskSegmentationHead = obj.MaskSegmentationHead.Learnables;
        end

        function obj = set.AllLearnables(obj, in)
            % Update Learnables on each subnetwork
            if(isfield(in, 'FeatureExtractionNet'))
                obj.FeatureExtractionNet.Learnables = in.FeatureExtractionNet;
            end

            if(isfield(in, 'RegionProposalNet'))
                obj.RegionProposalNet1.Learnables = in.RegionProposalNet1;
                obj.RegionProposalNet2.Learnables = in.RegionProposalNet2;
                obj.RegionProposalNet3.Learnables = in.RegionProposalNet3;

            end
            
            if(isfield(in, 'PostPoolFeatureExtractionNet'))
                obj.PostPoolFeatureExtractionNet.Learnables = in.PostPoolFeatureExtractionNet;
            end

            if(isfield(in, 'DetectionHeads'))
                obj.DetectionHeads.Learnables = in.DetectionHeads;
            end

            if(isfield(in, 'MaskSegmentationHead'))
                obj.MaskSegmentationHead.Learnables = in.MaskSegmentationHead;
            end

        end
    end

    methods(Access=protected)
        function obj = updateLearnableParameters(obj, updater)
            % Get the learnables, update them and then set them back on
            % the subnetworks.
            learnables = obj.AllLearnables;
            learnables = updater.apply(learnables);
            obj.AllLearnables = learnables;
        end
    end

    % getters for hidden properties needed by dltrain
    methods
    
        function layers = get.Layers(obj)

            layers = vertcat(obj.FeatureExtractionNet.Layers,...
                             obj.RegionProposalNet1.Layers,...
                             obj.RegionProposalNet2.Layers,...
                             obj.RegionProposalNet3.Layers,...
                             obj.PostPoolFeatureExtractionNet.Layers,...
                             obj.DetectionHeads.Layers,...
                             obj.MaskSegmentationHead.Layers);

        end
        
        function obj =  set.State(obj, value)

        end
        function state = get.State(obj)
            state = vertcat(obj.FeatureExtractionNet.State,...
                             obj.RegionProposalNet1.State,...
                             obj.RegionProposalNet2.State,...
                             obj.RegionProposalNet3.State,...
                             obj.PostPoolFeatureExtractionNet.State,...
                             obj.DetectionHeads.State,...
                             obj.MaskSegmentationHead.State);
        end

        function learnables = get.Learnables(obj)
            learnables = obj.AllLearnables;
        end

        function inputs = get.InputNames(obj)
            inputs = obj.FeatureExtractionNet.InputNames;
        end

        function outnames = get.OutputNames(obj)
            outnames = horzcat(...
                             obj.RegionProposalNet1.OutputNames,...
                             obj.RegionProposalNet2.OutputNames,...
                             obj.RegionProposalNet3.OutputNames,...
                             {'proposals'},...
                             obj.DetectionHeads.OutputNames,...
                             obj.MaskSegmentationHead.OutputNames);
        end

    end

end

% Helpers
%--------------------------------------------------------------------------
function classes = iGetCOCOClasses()
classes = {'person','bicycle','car','motorbike','aeroplane','bus',...
    'train','truck','boat','traffic light','fire hydrant',...
    'stop sign','parking meter','bench','bird','cat','dog',...
    'horse','sheep','cow','elephant','bear','zebra','giraffe',...
    'backpack','umbrella','handbag','tie','suitcase','frisbee',...
    'skis','snowboard','sports ball','kite','baseball bat',...
    'baseball glove','skateboard','surfboard','tennis racket',...
    'bottle','wine glass','cup','fork','knife','spoon','bowl',...
    'banana','apple','sandwich','orange','broccoli','carrot',...
    'hot dog','pizza','donut','cake','chair','sofa',...
    'pottedplant','bed','diningtable','toilet','tvmonitor',...
    'laptop','mouse','remote','keyboard','cell phone',...
    'microwave','oven','toaster','sink','refrigerator',...
    'book','clock','vase','scissors','teddy bear',...
    'hair drier','toothbrush'};
end

%--------------------------------------------------------------------------
function anchorBoxes = iGetDefaultCOCOAnchorBoxes()
baseAnchor = [11.313708305358887 22.627416610717773; 16 16; 22.627416610717773 11.313708305358887]*2;
anchorBoxes = round([baseAnchor; baseAnchor*2; baseAnchor*4; baseAnchor*8; baseAnchor*16]);
end

%--------------------------------------------------------------------------
function spkgroot = iTripwireMaskRCNN()
    % Check if support package is installed and return the installed
    % location
    breadcrumbFile = 'vision.internal.cnn.supportpackages.IsMaskRCNNInstalled';
    fullPath = which(breadcrumbFile);
    if isempty(fullPath)
        name     = 'Computer Vision Toolbox Model for Mask R-CNN Instance Segmentation';
        basecode = 'RCNN';
        throwAsCaller(MException(message('nnet_cnn:supportpackages:InstallRequired', mfilename, name, basecode)));
    end
    relPath =  fullfile(filesep,'+vision','+internal','+cnn',...
        '+supportpackages','IsMaskRCNNInstalled.m');
    idx = strfind(fullPath, relPath);
    spkgroot = fullPath(1:idx);
end

% Input validators
%--------------------------------------------------------------------------
function mustBeUniqueNames(input)    
     %Check for unique class names
     % input is a single char string, return
     if(ischar(input))
         return;
     end

     % For cell array of char strings, array of strings and categorical
     % arrays, check of uniqueness
     if(length(input)~=length(unique(input)))
        eid = 'vision:maskrcnn:duplicateClassNames';
        msg = 'The classNames must be unique.';
        throwAsCaller(MException(eid,msg))
     end
end

%--------------------------------------------------------------------------
function validateClassNames(input)
    
    % Check of single char string
    if(ischar(input))
        return;
    end
    
    % Check for cell array of char strings
    if(iscell(input))
        if(all(cellfun(@ischar, input)) && isvector(input))
            return;
        end
    end

    % Check for string & categorical vectors
    if( (isstring(input)||iscategorical(input)) && isvector(input))
        return;
    end

    eid = 'vision:maskrcnn:incorrectClassNameType';
        msg = ['The classNames should be one of the following types: cell array of char strings,' ...
            'string array or a categorical array.'];
        throwAsCaller(MException(eid,msg))

end

%--------------------------------------------------------------------------
function mustBeRGBSize(input)    
     % the size must either be [] or (1,3) with the channel dim =3
     if~(isempty(input)||(length(input)==3 && input(3)==3))
        eid = 'vision:maskrcnn:incorrectInputSize';
        msg = 'The input size must be 1x3 vector with the third element equal to 3.';
        throwAsCaller(MException(eid,msg))
     end
end

%--------------------------------------------------------------------------
function validateImageInput(in)
    
    im = [];
    if(isnumeric(in))
        im = in;
    elseif(matlab.io.datastore.internal.shim.isDatastore(in))
        out = preview(in);
        if(iscell(out))
            if(isempty(out))
                im = [];
            else
                im = out{1};
            end
        else
            im = out;
        end
    end

    if(~validateImage(im)||isempty(im))
        throwAsCaller(MException('vision:maskrcnn:invalidImageInput',...
               vision.getMessage('vision:maskrcnn:invalidImageInput')));
    end

end

%--------------------------------------------------------------------------
function tf = validateImage(in)
    tf = isnumeric(in)&&...
         ndims(in)<=4 && ... && numdims should be less than 3 
         (size(in,3)==3||size(in,3)==1); % gray scale or RGB image
end

%--------------------------------------------------------------------------
function validateLogicalFlag(in)
    validateattributes(in,{'logical'}, {'scalar','finite', 'real'});
end

%--------------------------------------------------------------------------
function iPrintHeader(printer)
    printer.printMessage('vision:maskrcnn:verboseHeader');
    printer.print('--------------------------');
    printer.linebreak();
end

%--------------------------------------------------------------------------
function updateMessage(printer, prevMessage, nextMessage)
    backspace = sprintf(repmat('\b',1,numel(prevMessage))); % figure how much to delete
    printer.print([backspace nextMessage]);
end

%--------------------------------------------------------------------------
function nextMessage = iPrintInitProgress(printer, prevMessage, k)
    nextMessage = getString(message('vision:maskrcnn:verboseProgressTxt',k));
    updateMessage(printer, prevMessage(1:end-1), nextMessage);
end

%--------------------------------------------------------------------------
function nextMessage = iPrintProgress(printer, prevMessage, k)
    nextMessage = getString(message('vision:maskrcnn:verboseProgressTxt',k));
    updateMessage(printer, prevMessage, nextMessage);
end

%--------------------------------------------------------------------------
function data = iLoadNetwork(spkgroot, datafolder, matfile)
matfile = fullfile(spkgroot, datafolder, matfile);
data = load(matfile);
end