function [net_stage4, net_stage5] = SEResNeXt101BackboneSplit(inputSize, cardinality, baseWidth)

lgraph_stage4 = layerGraph();
lgraph_stage5 = layerGraph();

% Input layer
inputLayers = [
    imageInputLayer(inputSize, 'Name', 'Input_data')
    convolution2dLayer([7 7], 64, 'Stride', 2, 'Padding', 3, 'Name', 'conv1', 'BiasInitializer','zeros','BiasLearnRateFactor',0)
    groupNormalizationLayer(min(32, 64 / 8), 'Name', 'gn1')  % Group Normalization
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(3, 'Stride', 2, 'Padding', 1, 'Name', 'maxpool1')
];

lgraph_stage4 = addLayers(lgraph_stage4, inputLayers);
lgraph_stage5 = addLayers(lgraph_stage5, inputLayers);

% Layer configurations for ResNeXt-101
configs = {
    struct('channels', 256, 'blocks', 3, 'stride', 1),  % Stage 2
    struct('channels', 512, 'blocks', 4, 'stride', 2),  % Stage 3
    struct('channels', 1024, 'blocks', 23, 'stride', 2), % Stage 4
    struct('channels', 2048, 'blocks', 3, 'stride', 2)  % Stage 5
};

currentInput = 'maxpool1';
currentChannels = 64;

% Add all residual stages
for stageIdx = 1:length(configs)
    config = configs{stageIdx};
    stageGraph = lgraph_stage4;
    if stageIdx == 4
        stageGraph = lgraph_stage5;
    end
    
    for blockIdx = 1:config.blocks
        stride = config.stride;
        if blockIdx > 1
            stride = 1;
        end
        
        baseName = sprintf('stage%d_block%d', stageIdx+1, blockIdx);
        width = cardinality * baseWidth;
        numGroups = min(32, width / 8);  % Ensure valid groups

        % Main path layers
        mainPath = [
            convolution2dLayer(1, width, 'Name', [baseName '_conv1'], 'Padding', 'same', 'BiasInitializer','zeros','BiasLearnRateFactor',0)
            groupNormalizationLayer(numGroups, 'Name', [baseName '_gn1'])
            reluLayer('Name', [baseName '_relu1'])
            
            groupedConvolution2dLayer(3, width, cardinality, 'Stride', stride, 'Name', [baseName '_conv2'], 'Padding', 'same', 'BiasInitializer','zeros','BiasLearnRateFactor',0)
            groupNormalizationLayer(numGroups, 'Name', [baseName '_gn2'])
            reluLayer('Name', [baseName '_relu2'])
            
            convolution2dLayer(1, config.channels, 'Name', [baseName '_conv3'], 'Padding', 'same', 'BiasInitializer','zeros','BiasLearnRateFactor',0)
            groupNormalizationLayer(numGroups, 'Name', [baseName '_gn3'])
            reluLayer('Name', [baseName '_relu_final'])
        ];
        
        stageGraph = addLayers(stageGraph, mainPath);
        
        if ismember(stageIdx, [1, 2, 3])
            lgraph_stage4 = stageGraph;
        else
            lgraph_stage5 = stageGraph;
        end
        
        % Only connect layers if they exist
        if isLayerPresent(stageGraph, currentInput) && isLayerPresent(stageGraph, [baseName '_conv1'])
            stageGraph = connectLayers(stageGraph, currentInput, [baseName '_conv1']);
        end
        
        currentInput = [baseName '_relu_final'];
        currentChannels = config.channels;
    end
end

% Initialize the networks
net_stage4 = dlnetwork(lgraph_stage4);
net_stage5 = dlnetwork(lgraph_stage5);
end

function exists = isLayerPresent(lgraph, layerName)
    exists = any(strcmp({lgraph.Layers.Name}, layerName));
end
