function net = createEfficientNetV2FeatureExtractor(inputSize)
    % Create EfficientNetV2-style network with GroupNorm that outputs a feature map
    % with 1/16 scale of the input and 1024 channels
    %
    % Parameters:
    %   inputSize - Input image size as [height width channels]
    %
    % Returns:
    %   net - The created dlnetwork
    
    if nargin < 1
        inputSize = [512 512 3]; % Default input size
    end
    
    lgraph = layerGraph();
    
    % Initial stem block with stride 2 (1/2 scale)
    stemLayers = [
        imageInputLayer(inputSize, 'Name', 'Input_data', 'Normalization', 'none')
        convolution2dLayer(3, 32, 'Stride', 2, 'Padding', 'same', 'Name', 'stem_conv', 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(calculateNumGroups(32), 'Name', 'stem_gn')
        swishLayer('Name', 'stem_swish')
    ];
    
    lgraph = addLayers(lgraph, stemLayers);
    
    % MBConv and Fused-MBConv block configurations
    % The goal is to reach 1/16 scale of the input
    % Starting from 1/2 after stem, we need 3 more stride-2 stages: 1/2 → 1/4 → 1/8 → 1/16
    blockConfigs = {
        % [block_type, channels, layers, stride, expansion]
        {'fused', 32,  2, 1, 1}    % Stage 1: maintain 1/2 scale
        {'fused', 64,  3, 2, 4}    % Stage 2: reduce to 1/4 scale
        {'mb',    128, 3, 2, 4}    % Stage 3: reduce to 1/8 scale
        {'mb',    256, 4, 2, 4}    % Stage 4: reduce to 1/16 scale
        {'mb',    512, 3, 1, 4}    % Stage 5: maintain 1/16 scale, increase channels
        {'mb',    1024, 1, 1, 2}   % Stage 6: maintain 1/16 scale, reach 1024 channels
    };
    
    prevLayerName = 'stem_swish';
    
    % Add all stages
    for stageIdx = 1:length(blockConfigs)
        config = blockConfigs{stageIdx};
        blockType = config{1};
        channels = config{2};
        numLayers = config{3};
        stride = config{4};
        expansion = config{5};
        
        for blockIdx = 1:numLayers
            currentStride = 1;
            if blockIdx == 1
                currentStride = stride;
            end
            
            if strcmp(blockType, 'fused')
                [lgraph, prevLayerName] = addFusedMBConvBlock(lgraph, prevLayerName, ...
                    channels, currentStride, expansion, ...
                    sprintf('stage%d_block%d', stageIdx, blockIdx));
            else
                [lgraph, prevLayerName] = addMBConvBlock(lgraph, prevLayerName, ...
                    channels, currentStride, expansion, ...
                    sprintf('stage%d_block%d', stageIdx, blockIdx));
            end
        end
    end
    
    % The output is already at 1/16 scale with 1024 channels from the last block
    % We'll add a final normalization to ensure good feature quality
    finalLayers = [
        swishLayer('Name', 'feature_out')
    ];
    
    lgraph = addLayers(lgraph, finalLayers);
    lgraph = connectLayers(lgraph, prevLayerName, 'feature_out');
    
    % Convert to dlnetwork
    net = dlnetwork(lgraph);
    
    % All convolutional layers already have zero biases
end

function [lgraph, lastLayerName] = addFusedMBConvBlock(lgraph, inputName, channels, stride, expansion, blockName)
    % Fused MBConv block implementation with GroupNorm
    expanded = channels * expansion;
    
    layers = [
        convolution2dLayer(3, expanded, 'Stride', stride, 'Padding', 'same', ...
            'Name', [blockName '_fused_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(calculateNumGroups(expanded), 'Name', [blockName '_gn1'])
        swishLayer('Name', [blockName '_swish1'])
    ];
    
    if expanded ~= channels
        layers = [layers
            convolution2dLayer(1, channels, 'Stride', 1, 'Padding', 'same', ...
                'Name', [blockName '_project_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
            groupNormalizationLayer(calculateNumGroups(channels), 'Name', [blockName '_gn2'])
        ];
    end
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, inputName, [blockName '_fused_conv']);
    
    lastLayerName = layers(end).Name;
    
    % Add residual connection if possible
    if stride == 1 && ~strcmp(inputName, 'stem_swish')
        % Check if channels match for residual connection
        inputLayer = findLayerByName(lgraph, inputName);
        inputChannels = getChannelsByLayerName(lgraph, inputName);
        
        if inputChannels == channels
            lgraph = addLayers(lgraph, additionLayer(2, 'Name', [blockName '_add']));
            lgraph = connectLayers(lgraph, inputName, [blockName '_add/in1']);
            lgraph = connectLayers(lgraph, lastLayerName, [blockName '_add/in2']);
            lastLayerName = [blockName '_add'];
        end
    end
end

function [lgraph, lastLayerName] = addMBConvBlock(lgraph, inputName, channels, stride, expansion, blockName)
    % MBConv block implementation with GroupNorm
    expanded = channels * expansion;
    
    layers = [
        convolution2dLayer(1, expanded, 'Stride', 1, 'Padding', 'same', ...
            'Name', [blockName '_expand_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(calculateNumGroups(expanded), 'Name', [blockName '_gn1'])
        swishLayer('Name', [blockName '_swish1'])
        
        groupedConvolution2dLayer(3, 1, 'channel-wise', 'Stride', stride, ...
            'Padding', 'same', 'Name', [blockName '_depthwise_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(calculateNumGroups(expanded), 'Name', [blockName '_gn2'])
        swishLayer('Name', [blockName '_swish2'])
        
        convolution2dLayer(1, channels, 'Stride', 1, 'Padding', 'same', ...
            'Name', [blockName '_project_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(calculateNumGroups(channels), 'Name', [blockName '_gn3'])
    ];
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, inputName, [blockName '_expand_conv']);
    
    lastLayerName = layers(end).Name;
    
    % Add residual connection if possible
    if stride == 1 && ~strcmp(inputName, 'stem_swish')
        % Check if channels match for residual connection
        inputChannels = getChannelsByLayerName(lgraph, inputName);
        
        if inputChannels == channels
            lgraph = addLayers(lgraph, additionLayer(2, 'Name', [blockName '_add']));
            lgraph = connectLayers(lgraph, inputName, [blockName '_add/in1']);
            lgraph = connectLayers(lgraph, lastLayerName, [blockName '_add/in2']);
            lastLayerName = [blockName '_add'];
        end
    end
end

% Function removed as we're setting zero biases directly during layer creation

function numGroups = calculateNumGroups(numChannels)
    % Calculate number of groups based on requirements:
    % - Minimum 16 channels per group (adjusted for better division)
    % - Maximum 32 groups
    numGroups = min(floor(numChannels / 16), 32);
    
    % Ensure at least 1 group
    numGroups = max(numGroups, 1);
    
    % Ensure numChannels is divisible by numGroups
    while mod(numChannels, numGroups) ~= 0 && numGroups > 1
        numGroups = numGroups - 1;
    end
end

function idx = findLayerIndex(layers, layerName)
    % Find the index of a layer by its name
    idx = -1;
    for i = 1:length(layers)
        if strcmp(layers(i).Name, layerName)
            idx = i;
            return;
        end
    end
end

function channels = getChannelsByLayerName(lgraph, layerName)
    % Get the number of channels of a layer by its name
    layer = findLayerByName(lgraph, layerName);
    
    if isa(layer, 'nnet.cnn.layer.ImageInputLayer')
        channels = layer.InputSize(3);
    elseif isa(layer, 'nnet.cnn.layer.Convolution2DLayer') || ...
           isa(layer, 'nnet.cnn.layer.GroupedConvolution2DLayer')
        channels = layer.NumFilters;
    elseif isa(layer, 'nnet.cnn.layer.AdditionLayer') || ...
           isa(layer, 'nnet.cnn.layer.GroupNormalizationLayer') || ...
           isa(layer, 'nnet.cnn.layer.ReLULayer') || ...
           isa(layer, 'nnet.cnn.layer.SoftmaxLayer') || ...
           isa(layer, 'function_handle')
        % For these layers, we need to check the preceding layer
        incomingLayers = findIncomingLayers(lgraph, layerName);
        if ~isempty(incomingLayers)
            channels = getChannelsByLayerName(lgraph, incomingLayers{1});
        else
            channels = 0;
        end
    else
        % For unsupported layer types
        channels = 0;
    end
end

function layer = findLayerByName(lgraph, layerName)
    % Find a layer in the layer graph by its name
    for i = 1:length(lgraph.Layers)
        if strcmp(lgraph.Layers(i).Name, layerName)
            layer = lgraph.Layers(i);
            return;
        end
    end
    layer = [];
end

function incomingLayers = findIncomingLayers(lgraph, layerName)
    % Find all layers that connect to the specified layer
    incomingLayers = {};
    for i = 1:size(lgraph.Connections, 1)
        if strcmp(lgraph.Connections.Destination{i}, layerName)
            incomingLayers{end+1} = lgraph.Connections.Source{i};
        end
    end
end

% Using MATLAB's built-in swishLayer