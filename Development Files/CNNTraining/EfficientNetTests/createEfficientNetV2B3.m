function net = createEfficientNetV2B3()
    % Create EfficientNetV2-B3 network with GroupNorm instead of BatchNorm
    lgraph = layerGraph();
    
    % Initial stem block
    stemLayers = [
        imageInputLayer([528 704 1], 'Name', 'input')
        convolution2dLayer(3, 40, 'Stride', 2, 'Padding', 'same', 'Name', 'stem_conv')
        groupNormalizationLayer(calculateNumGroups(40), 'Name', 'stem_gn')
        swishLayer('Name', 'stem_swish')
    ];
    
    lgraph = addLayers(lgraph, stemLayers);
    
    % MBConv and Fused-MBConv block configurations for B3
    blockConfigs = {
        % [block_type, channels, layers, stride, expansion]
        {'fused', 40,  2, 1, 1}    % Stage 1
        {'fused', 64,  3, 2, 1}    % Stage 2
        {'mb',    80,  3, 2, 4}    % Stage 3
        {'mb',    112, 4, 2, 4}    % Stage 4
        {'mb',    192, 4, 2, 4}    % Stage 5
        {'mb',    256, 6, 2, 6}    % Stage 6
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
    
    % Top layers
    topLayers = [
        convolution2dLayer(1, 1280, 'Name', 'top_conv')
        groupNormalizationLayer(calculateNumGroups(1280), 'Name', 'top_gn')
        swishLayer('Name', 'top_swish')
        globalAveragePooling2dLayer('Name', 'gap')
        dropoutLayer(0.3, 'Name', 'dropout')
        fullyConnectedLayer(1000, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
    ];
    
    lgraph = addLayers(lgraph, topLayers);
    lgraph = connectLayers(lgraph, prevLayerName, 'top_conv');
    
    % Convert to dlnetwork
    net = dlnetwork(lgraph);
end

function [lgraph, lastLayerName] = addFusedMBConvBlock(lgraph, inputName, channels, stride, expansion, blockName)
    % Fused MBConv block implementation with GroupNorm
    expanded = channels * expansion;
    
    layers = [
        convolution2dLayer(3, expanded, 'Stride', stride, 'Padding', 'same', ...
            'Name', [blockName '_fused_conv'])
        groupNormalizationLayer(calculateNumGroups(expanded), 'Name', [blockName '_gn1'])
        swishLayer('Name', [blockName '_swish1'])
    ];
    
    if expanded ~= channels
        layers = [layers
            convolution2dLayer(1, channels, 'Stride', 1, 'Padding', 'same', ...
                'Name', [blockName '_project_conv'])
            groupNormalizationLayer(calculateNumGroups(channels), 'Name', [blockName '_gn2'])
        ];
    end
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, inputName, [blockName '_fused_conv']);
    
    lastLayerName = layers(end).Name;
    
    % Add residual connection if possible
    if stride == 1 && strcmp(inputName, 'stem_swish') == 0
        lgraph = addLayers(lgraph, additionLayer(2, 'Name', [blockName '_add']));
        lgraph = connectLayers(lgraph, inputName, [blockName '_add/in1']);
        lgraph = connectLayers(lgraph, lastLayerName, [blockName '_add/in2']);
        lastLayerName = [blockName '_add'];
    end
end

function [lgraph, lastLayerName] = addMBConvBlock(lgraph, inputName, channels, stride, expansion, blockName)
    % MBConv block implementation with GroupNorm
    expanded = channels * expansion;
    
    layers = [
        convolution2dLayer(1, expanded, 'Stride', 1, 'Padding', 'same', ...
            'Name', [blockName '_expand_conv'])
        groupNormalizationLayer(calculateNumGroups(expanded), 'Name', [blockName '_gn1'])
        swishLayer('Name', [blockName '_swish1'])
        
        groupedConvolution2dLayer(3, expanded, expanded, 'Stride', stride, ...
            'Padding', 'same', 'Name', [blockName '_depthwise_conv'])
        groupNormalizationLayer(calculateNumGroups(expanded), 'Name', [blockName '_gn2'])
        swishLayer('Name', [blockName '_swish2'])
        
        convolution2dLayer(1, channels, 'Stride', 1, 'Padding', 'same', ...
            'Name', [blockName '_project_conv'])
        groupNormalizationLayer(calculateNumGroups(channels), 'Name', [blockName '_gn3'])
    ];
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, inputName, [blockName '_expand_conv']);
    
    lastLayerName = layers(end).Name;
    
    % Add residual connection if possible
    if stride == 1 && strcmp(inputName, 'stem_swish') == 0
        lgraph = addLayers(lgraph, additionLayer(2, 'Name', [blockName '_add']));
        lgraph = connectLayers(lgraph, inputName, [blockName '_add/in1']);
        lgraph = connectLayers(lgraph, lastLayerName, [blockName '_add/in2']);
        lastLayerName = [blockName '_add'];
    end
end

function numGroups = calculateNumGroups(numChannels)
    % Calculate number of groups based on requirements:
    % - Minimum 32 channels per group
    % - Maximum 32 groups
    numGroups = min(floor(numChannels / 32), 32);
    
    % Ensure at least 1 group
    numGroups = max(numGroups, 1);
    
    % Ensure numChannels is divisible by numGroups
    while mod(numChannels, numGroups) ~= 0 && numGroups > 1
        numGroups = numGroups - 1;
    end
end