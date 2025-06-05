function net = createEfficientNetV2FeatureExtractor(inputSize)
    % Create EfficientNetV2-style network with GroupNorm that outputs a feature map
    % with 1/16 scale of the input and 1024 channels

    if nargin < 1
        inputSize = [512 512 3]; % Default input size if none is provided
    end

    lgraph = layerGraph(); % Initialize empty layer graph

    % Initial stem block with stride 2 (reduces spatial resolution by half)
    stemLayers = [
        imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none') % Input layer
        convolution2dLayer(3, 32, 'Stride', 2, 'Padding', 'same', 'Name', 'stem_conv', 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros') % Conv layer with downsampling
        groupNormalizationLayer(calculateNumGroups(32), 'Name', 'stem_gn') % Group normalization
        swishLayer('Name', 'stem_swish') % Swish activation
    ];

    lgraph = addLayers(lgraph, stemLayers); % Add stem block to graph

    % Configuration for each stage: {block type, output channels, number of layers, stride, expansion factor}
    blockConfigs = {
        {'fused', 32,  2, 1, 1}
        {'fused', 64,  3, 2, 4}
        {'mb',    128, 3, 2, 4}
        {'mb',    256, 4, 2, 4}
        {'mb',    512, 3, 1, 4}
        {'mb',    1024, 1, 1, 2}
    };

    prevLayerName = 'stem_swish'; % Track last layer name for connections

    % Iterate through stages and add blocks to graph
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
                currentStride = stride; % Apply stride on first block of each stage
            end

            % Add appropriate block type
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

    % Final normalization and activation
    finalLayers = [
        groupNormalizationLayer(calculateNumGroups(1024), 'Name', 'final_gn')
        swishLayer('Name', 'final_swish')
    ];

    lgraph = addLayers(lgraph, finalLayers); % Add final layers
    lgraph = connectLayers(lgraph, prevLayerName, 'final_gn'); % Connect last block to final layers

    net = dlnetwork(lgraph); % Convert to dlnetwork object
end

function [lgraph, lastLayerName] = addFusedMBConvBlock(lgraph, inputName, channels, stride, expansion, blockName)
    % Add Fused-MBConv block with optional projection and residual connection
    expanded = channels * expansion;

    % Main fused convolution layer with Swish activation
    layers = [
        convolution2dLayer(3, expanded, 'Stride', stride, 'Padding', 'same', ...
            'Name', [blockName '_fused_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(calculateNumGroups(expanded), 'Name', [blockName '_gn1'])
        swishLayer('Name', [blockName '_swish1'])
    ];

    % Optional projection layer (1x1 conv) if expansion != 1
    if expanded ~= channels
        layers = [layers
            convolution2dLayer(1, channels, 'Stride', 1, 'Padding', 'same', ...
                'Name', [blockName '_project_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
            groupNormalizationLayer(calculateNumGroups(channels), 'Name', [blockName '_gn2'])
        ];
    end

    lgraph = addLayers(lgraph, layers); % Add layers to graph
    lgraph = connectLayers(lgraph, inputName, [blockName '_fused_conv']); % Connect input

    lastLayerName = layers(end).Name; % Track last layer for connections

    % Add residual connection if applicable
    if stride == 1 && ~strcmp(inputName, 'stem_swish')
        inputChannels = getChannelsByLayerName(lgraph, inputName);
        if inputChannels == channels
            lgraph = addLayers(lgraph, additionLayer(2, 'Name', [blockName '_add']));
            lgraph = connectLayers(lgraph, inputName, [blockName '_add/in1']);
            lgraph = connectLayers(lgraph, lastLayerName, [blockName '_add/in2']);
            lastLayerName = [blockName '_add']; % Update last layer to addition
        end
    end
end

function [lgraph, lastLayerName] = addMBConvBlock(lgraph, inputName, channels, stride, expansion, blockName)
    % Add MBConv block (expand -> depthwise -> project) with residual

    expanded = channels * expansion;

    % Expand (1x1), depthwise (3x3), project (1x1) convs
    expandConv = convolution2dLayer(1, expanded, 'Stride', 1, 'Padding', 'same', ...
        'Name', [blockName '_expand_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros');
    gn1 = groupNormalizationLayer(calculateNumGroups(expanded), 'Name', [blockName '_gn1']);
    sw1 = swishLayer('Name', [blockName '_swish1']);

    dwConv = groupedConvolution2dLayer(3, expanded, expanded, 'Stride', stride, 'Padding', 'same', ...
        'Name', [blockName '_depthwise_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros');
    gn2 = groupNormalizationLayer(calculateNumGroups(expanded), 'Name', [blockName '_gn2']);
    sw2 = swishLayer('Name', [blockName '_swish2']);

    projConv = convolution2dLayer(1, channels, 'Stride', 1, 'Padding', 'same', ...
        'Name', [blockName '_project_conv'], 'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros');
    gn3 = groupNormalizationLayer(calculateNumGroups(channels), 'Name', [blockName '_gn3']);

    layers = [expandConv; gn1; sw1; dwConv; gn2; sw2; projConv; gn3];

    lgraph = addLayers(lgraph, layers); % Add MBConv block layers
    lgraph = connectLayers(lgraph, inputName, expandConv.Name); % Connect input

    lastLayerName = gn3.Name;

    % Residual connection
    if stride == 1 && ~strcmp(inputName, 'stem_swish')
        inputChannels = getChannelsByLayerName(lgraph, inputName);
        if inputChannels == channels
            lgraph = addLayers(lgraph, additionLayer(2, 'Name', [blockName '_add']));
            lgraph = connectLayers(lgraph, inputName, [blockName '_add/in1']);
            lgraph = connectLayers(lgraph, lastLayerName, [blockName '_add/in2']);
            lastLayerName = [blockName '_add'];
        end
    end
end

function numGroups = calculateNumGroups(numChannels)
    % Calculate suitable number of groups for group normalization
    numGroups = min(floor(numChannels / 16), 32);
    numGroups = max(numGroups, 1);
    while mod(numChannels, numGroups) ~= 0 && numGroups > 1
        numGroups = numGroups - 1;
    end
end

function layer = findLayerByName(lgraph, layerName)
    % Search for a layer in the layer graph by name
    for i = 1:length(lgraph.Layers)
        if strcmp(lgraph.Layers(i).Name, layerName)
            layer = lgraph.Layers(i);
            return;
        end
    end
    layer = [];
end

function channels = getChannelsByLayerName(lgraph, layerName)
    % Get number of output channels from a layer
    layer = findLayerByName(lgraph, layerName);

    if isa(layer, 'nnet.cnn.layer.ImageInputLayer')
        channels = layer.InputSize(3);
    elseif isa(layer, 'nnet.cnn.layer.Convolution2DLayer') || ...
           isa(layer, 'nnet.cnn.layer.GroupedConvolution2DLayer') || ...
           isa(layer, 'nnet.cnn.layer.DepthwiseConvolution2DLayer')
        channels = layer.NumFilters;
    else
        % Fallback: recursively check incoming layer
        incomingLayers = findIncomingLayers(lgraph, layerName);
        if ~isempty(incomingLayers)
            channels = getChannelsByLayerName(lgraph, incomingLayers{1});
        else
            channels = 0;
        end
    end
end

function incomingLayers = findIncomingLayers(lgraph, layerName)
    % Return names of layers connected to the input of the given layer
    incomingLayers = {};
    for i = 1:size(lgraph.Connections, 1)
        if strcmp(lgraph.Connections.Destination{i}, layerName)
            incomingLayers{end+1} = lgraph.Connections.Source{i};
        end
    end
end
