function net = createEfficientNetV2Decoder(inputSize)
    % Creates a decoder network that upsamples from 1/16 scale with 1024 channels
    % to 1/2 scale with 2048 channels
    %
    % Parameters:
    %   inputSize - Input feature map size as [height width channels]
    %               Default is [32 32 1024] (assuming 512x512 original image)
    %
    % Returns:
    %   net - The created dlnetwork
    
    if nargin < 1
        inputSize = [32 32 1024]; % Default for 512x512 input to the encoder
    end
    
    lgraph = layerGraph();
    
    % Input layer for the feature map
    inputLayer = imageInputLayer(inputSize, 'Name', 'decoder_input');
    lgraph = addLayers(lgraph, inputLayer);
    
    prevLayerName = 'decoder_input';
    
    % Stage 1: Maintain 1/16 scale, convert 1024 → 1024 channels
    [lgraph, prevLayerName] = addInvMBConvBlock(lgraph, prevLayerName, ...
        1024, 1, 4, 'decoder_stage1_block1');
    
    % Stage 2: Upsample 1/16 → 1/8 scale, convert 1024 → 768 channels
    [lgraph, prevLayerName] = addUpsampleStage(lgraph, prevLayerName, ...
        768, 2, 'decoder_stage2');
    
    % Stage 3: Upsample 1/8 → 1/4 scale, convert 768 → 512 channels
    [lgraph, prevLayerName] = addUpsampleStage(lgraph, prevLayerName, ...
        512, 2, 'decoder_stage3');
    
    % Stage 4: Upsample 1/4 → 1/2 scale, convert 512 → 2048 channels
    [lgraph, prevLayerName] = addUpsampleStage(lgraph, prevLayerName, ...
        2048, 2, 'decoder_stage4');
    
    % Final normalization and activation
    finalLayers = [
        groupNormalizationLayer(calculateNumGroups(2048), 'Name', 'final_gn')
        swishLayer('Name', 'final_swish')
    ];
    
    lgraph = addLayers(lgraph, finalLayers);
    lgraph = connectLayers(lgraph, prevLayerName, 'final_gn');
    
    % Convert to dlnetwork
    net = dlnetwork(lgraph);
end

function [lgraph, lastLayerName] = addUpsampleStage(lgraph, inputName, outChannels, scale, stageName)
    % Add an upsampling stage with a transpose convolution followed by MBConv blocks
    
    % First determine input channels
    inputChannels = getChannelsByLayerName(lgraph, inputName);
    
    % Add transpose convolution for upsampling
    upsampleLayers = [
        transposedConv2dLayer(2, outChannels, 'Stride', scale, 'Cropping', 'same', ...
            'Name', [stageName '_upsample'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(calculateNumGroups(outChannels), 'Name', [stageName '_upsample_gn'])
        swishLayer('Name', [stageName '_upsample_swish'])
    ];
    
    lgraph = addLayers(lgraph, upsampleLayers);
    lgraph = connectLayers(lgraph, inputName, [stageName '_upsample']);
    
    % Add two MB Conv blocks to refine features
    [lgraph, lastLayerName] = addInvMBConvBlock(lgraph, [stageName '_upsample_swish'], ...
        outChannels, 1, 4, [stageName '_block1']);
    
    [lgraph, lastLayerName] = addInvMBConvBlock(lgraph, lastLayerName, ...
        outChannels, 1, 4, [stageName '_block2']);
end

function [lgraph, lastLayerName] = addInvMBConvBlock(lgraph, inputName, channels, stride, expansion, blockName)
    % Inverted MBConv block similar to the one in the encoder but with GroupNorm
    % This maintains the EfficientNet-style architecture
    
    % Get input channels
    inputChannels = getChannelsByLayerName(lgraph, inputName);
    expanded = channels * expansion;
    
    layers = [
        % Expand to higher dimension first
        convolution2dLayer(1, expanded, 'Stride', 1, 'Padding', 'same', ...
            'Name', [blockName '_expand_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(calculateNumGroups(expanded), 'Name', [blockName '_gn1'])
        swishLayer('Name', [blockName '_swish1'])
        
        % Depthwise convolution
        groupedConvolution2dLayer(3, 1, 'channel-wise', 'Stride', stride, ...
            'Padding', 'same', 'Name', [blockName '_depthwise_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(calculateNumGroups(expanded), 'Name', [blockName '_gn2'])
        swishLayer('Name', [blockName '_swish2'])
        
        % Project back to output channels
        convolution2dLayer(1, channels, 'Stride', 1, 'Padding', 'same', ...
            'Name', [blockName '_project_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(calculateNumGroups(channels), 'Name', [blockName '_gn3'])
    ];
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, inputName, [blockName '_expand_conv']);
    
    lastLayerName = layers(end).Name;
    
    % Add residual connection if possible (same channels and no stride)
    if stride == 1 && inputChannels == channels
        lgraph = addLayers(lgraph, additionLayer(2, 'Name', [blockName '_add']));
        lgraph = connectLayers(lgraph, inputName, [blockName '_add/in1']);
        lgraph = connectLayers(lgraph, lastLayerName, [blockName '_add/in2']);
        lastLayerName = [blockName '_add'];
    end
end

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

function channels = getChannelsByLayerName(lgraph, layerName)
    % Get the number of channels of a layer by its name
    layer = findLayerByName(lgraph, layerName);
    
    if isa(layer, 'nnet.cnn.layer.ImageInputLayer') || isa(layer, 'nnet.cnn.layer.FeatureInputLayer')
        channels = layer.InputSize(end);
    elseif isa(layer, 'nnet.cnn.layer.Convolution2DLayer') || ...
           isa(layer, 'nnet.cnn.layer.GroupedConvolution2DLayer') || ...
           isa(layer, 'nnet.cnn.layer.TransposedConv2DLayer')
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