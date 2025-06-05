function net = createEfficientNetDownsamplingTail()
    % EfficientNet-style downsampling tail block with residual connection
    % Now enhanced with Squeeze-and-Excitation (SE) block
    % Input: 14x14x1024 → Output: 7x7x2048

    lgraph = layerGraph();

    blockName = 'tail_block';
    inChannels = 1024;
    outChannels = 2048;
    expansion = 2;
    expanded = inChannels * expansion;
    seRatio = 0.25;  % Added SE ratio parameter

    % Calculate numGroups for GroupNorm
    numGroupsExpanded = min(floor(expanded / 32), 32);
    numGroupsExpanded = max(numGroupsExpanded, 1);
    while mod(expanded, numGroupsExpanded) ~= 0 && numGroupsExpanded > 1
        numGroupsExpanded = numGroupsExpanded - 1;
    end
    numGroupsOut = min(floor(outChannels / 32), 32);
    numGroupsOut = max(numGroupsOut, 1);
    while mod(outChannels, numGroupsOut) ~= 0 && numGroupsOut > 1
        numGroupsOut = numGroupsOut - 1;
    end

    % Input layer for 14x14x1024 input
    inputLayer = imageInputLayer([14 14 inChannels], 'Name', 'input', 'Normalization', 'none');
    lgraph = addLayers(lgraph, inputLayer);

    %% Main path layers
    mainLayers = [
        convolution2dLayer(1, expanded, 'Stride', 1, 'Padding', 'same', ...
            'Name', [blockName '_expand_conv'], ...
            'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(numGroupsExpanded, 'Name', [blockName '_gn1'])
        swishLayer('Name', [blockName '_swish1'])

        groupedConvolution2dLayer(3, 1, expanded, 'Stride', 2, ...
            'Padding', 'same', 'Name', [blockName '_depthwise_conv'])
        groupNormalizationLayer(numGroupsExpanded, 'Name', [blockName '_gn2'])
        swishLayer('Name', [blockName '_swish2'])
    ];
    lgraph = addLayers(lgraph, mainLayers);
    lgraph = connectLayers(lgraph, 'input', [blockName '_expand_conv']);

    %% Add Squeeze-and-Excitation block after depthwise conv
    if seRatio > 0
        [lgraph, seOutName] = addSqueezeExcitationBlock(lgraph, [blockName '_swish2'], expanded, seRatio, blockName);
        lastLayerAfterSE = seOutName;
    else
        lastLayerAfterSE = [blockName '_swish2'];
    end

    %% Project path (1x1 conv)
    projectLayers = [
        convolution2dLayer(1, outChannels, 'Stride', 1, 'Padding', 'same', ...
            'Name', [blockName '_project_conv'], ...
            'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(numGroupsOut, 'Name', [blockName '_gn3'])
    ];
    lgraph = addLayers(lgraph, projectLayers);
    lgraph = connectLayers(lgraph, lastLayerAfterSE, [blockName '_project_conv']);

    %% Shortcut projection path
    shortcutLayers = [
        convolution2dLayer(1, outChannels, 'Stride', 2, 'Padding', 'same', ...
            'Name', [blockName '_shortcut_conv'], ...
            'BiasLearnRateFactor', 0, 'BiasInitializer', 'zeros')
        groupNormalizationLayer(numGroupsOut, 'Name', [blockName '_shortcut_gn'])
    ];
    lgraph = addLayers(lgraph, shortcutLayers);
    lgraph = connectLayers(lgraph, 'input', [blockName '_shortcut_conv']);

    %% Addition + final Swish activation
    addition = additionLayer(2, 'Name', [blockName '_add']);
    finalSwish = swishLayer('Name', [blockName '_swish_out']);
    lgraph = addLayers(lgraph, [addition, finalSwish]);

    %% Connections
    % Connect main path
    lgraph = connectLayers(lgraph, [blockName '_gn3'], [blockName '_add/in1']);

    % Connect shortcut path
    lgraph = connectLayers(lgraph, [blockName '_shortcut_gn'], [blockName '_add/in2']);

    %% Final dlnetwork
    exampleInput = dlarray(zeros(14, 14, 1024, 1), 'SSCB');
    net = dlnetwork(lgraph, 'Initialize', true);
end

function [lgraph, outputName] = addSqueezeExcitationBlock(lgraph, inputName, channels, seRatio, blockName)
    % Add a Squeeze-and-Excitation block
    % seRatio determines the reduction ratio for the bottleneck
    
    % Calculate the number of bottleneck channels
    bottleneckChannels = max(1, floor(channels * seRatio));
    
    % Create a custom SE block using MATLAB's layers with proper reshaping
    % for compatibility with spatial features
    
    % 1. Global Average Pooling to get channel statistics
    poolLayer = globalAveragePooling2dLayer('Name', [blockName '_se_pool']);
    
    % 2. Fully connected layers for channel attention
    fc1 = convolution2dLayer(1, bottleneckChannels, 'Name', [blockName '_se_fc1']);
    swish1 = swishLayer('Name', [blockName '_se_swish']);
    fc2 = convolution2dLayer(1, channels, 'Name', [blockName '_se_fc2']);
    sigmoid1 = sigmoidLayer('Name', [blockName '_se_sigmoid']);
    
    % 3. Custom function layer to reshape and apply weights to input feature map
    seApplyLayer = multiplicationLayer(2, ...
                                'Name', [blockName '_se_apply']);
    
    % Add all layers to graph
    lgraph = addLayers(lgraph, poolLayer);
    lgraph = connectLayers(lgraph, inputName, poolLayer.Name);
    
    lgraph = addLayers(lgraph, fc1);
    lgraph = connectLayers(lgraph, poolLayer.Name, fc1.Name);
    
    lgraph = addLayers(lgraph, swish1);
    lgraph = connectLayers(lgraph, fc1.Name, swish1.Name);
    
    lgraph = addLayers(lgraph, fc2);
    lgraph = connectLayers(lgraph, swish1.Name, fc2.Name);
    
    lgraph = addLayers(lgraph, sigmoid1);
    lgraph = connectLayers(lgraph, fc2.Name, sigmoid1.Name);
    
    lgraph = addLayers(lgraph, seApplyLayer);
    lgraph = connectLayers(lgraph, sigmoid1.Name, [blockName '_se_apply/in1']);
    lgraph = connectLayers(lgraph, inputName, [blockName '_se_apply/in2']);
    
    outputName = [blockName '_se_apply'];
end

function output = applyChannelAttention(weights, featureMap)
    % This function applies channel attention weights to the input feature map
    % weights: output from sigmoid (batch_size x num_channels)
    % featureMap: input feature map (height x width x channels x batch_size)
    
    % Get dimensions
    [h, w, c, n] = size(featureMap);
    
    % Apply attention via multiplication
    output = featureMap .* weights;
end