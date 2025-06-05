clear
clc

% Define network parameters
inputSize = [224 224 3];
numClasses = 1000;
cardinality = 32;
baseWidth = 8;

lgraph = layerGraph();

% Input layer
tempLayers = [
    imageInputLayer(inputSize, 'Name', 'input')
    convolution2dLayer([7 7], 64, 'Stride', 2, 'Padding', 3, 'Name', 'conv1')
    groupNormalizationLayer(32, 'Name', 'gn1')  % Group Normalization
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(3, 'Stride', 2, 'Padding', 1, 'Name', 'maxpool1')];

lgraph = addLayers(lgraph, tempLayers);

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
    
    for blockIdx = 1:config.blocks
        stride = config.stride;
        if blockIdx > 1
            stride = 1;
        end
        
        baseName = sprintf('stage%d_block%d', stageIdx+1, blockIdx);
        width = cardinality * baseWidth;
        
        % Main path layers
        mainPath = [
            convolution2dLayer(1, width, 'Name', [baseName '_conv1'], 'Padding', 'same')
            groupNormalizationLayer(32, 'Name', [baseName '_gn1'])  % Group Norm
            reluLayer('Name', [baseName '_relu1'])
            
            groupedConvolution2dLayer(3, width, cardinality, 'Stride', stride, ...
                'Name', [baseName '_conv2'], 'Padding', 'same')
            groupNormalizationLayer(32, 'Name', [baseName '_gn2'])  % Group Norm
            reluLayer('Name', [baseName '_relu2'])
            
            convolution2dLayer(1, config.channels, 'Name', [baseName '_conv3'], 'Padding', 'same')
            groupNormalizationLayer(32, 'Name', [baseName '_gn3'])];  % Group Norm
        
        % SE block (using 1x1 convolutions instead of FC layers)
        sePath = [
            globalAveragePooling2dLayer('Name', [baseName '_gap'])
            
            convolution2dLayer(1, round(config.channels / 16), 'Name', [baseName '_conv1x1_1'])
            reluLayer('Name', [baseName '_relu_se'])
            
            convolution2dLayer(1, config.channels, 'Name', [baseName '_conv1x1_2'])
            sigmoidLayer('Name', [baseName '_sigmoid'])];
        
        % Add layers to graph
        lgraph = addLayers(lgraph, mainPath);
        lgraph = addLayers(lgraph, sePath);
        
        % Connect main path
        lgraph = connectLayers(lgraph, currentInput, [baseName '_conv1']);
        
        % Add shortcut if needed
        if stride ~= 1 || currentChannels ~= config.channels
            shortcut = [
                convolution2dLayer(1, config.channels, 'Stride', stride, ...
                    'Name', [baseName '_shortcut'])
                groupNormalizationLayer(32, 'Name', [baseName '_shortcut_gn'])];  % Group Norm
            lgraph = addLayers(lgraph, shortcut);
            lgraph = connectLayers(lgraph, currentInput, [baseName '_shortcut']);
        end
        
        % Multiplication and Addition layers for SE block
        lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', [baseName '_multiply']));
        lgraph = addLayers(lgraph, additionLayer(2, 'Name', [baseName '_add']));
        
        % Connect SE block
        lgraph = connectLayers(lgraph, [baseName '_gn3'], [baseName '_gap']);
if ~isConnectionPresent(lgraph, [baseName '_gap'], [baseName '_conv1x1_1'])
    lgraph = connectLayers(lgraph, [baseName '_gap'], [baseName '_conv1x1_1']);
end

if ~isConnectionPresent(lgraph, [baseName '_conv1x1_1'], [baseName '_relu_se'])
    lgraph = connectLayers(lgraph, [baseName '_conv1x1_1'], [baseName '_relu_se']);
end

if ~isConnectionPresent(lgraph, [baseName '_relu_se'], [baseName '_conv1x1_2'])
    lgraph = connectLayers(lgraph, [baseName '_relu_se'], [baseName '_conv1x1_2']);
end

if ~isConnectionPresent(lgraph, [baseName '_conv1x1_2'], [baseName '_sigmoid'])
    lgraph = connectLayers(lgraph, [baseName '_conv1x1_2'], [baseName '_sigmoid']);
end

if ~isConnectionPresent(lgraph, [baseName '_sigmoid'], [baseName '_multiply/in2'])
    lgraph = connectLayers(lgraph, [baseName '_sigmoid'], [baseName '_multiply/in2']);
end

        lgraph = connectLayers(lgraph, [baseName '_gn3'], [baseName '_multiply/in1']);
        
        % Connect addition layer
        lgraph = connectLayers(lgraph, [baseName '_multiply'], [baseName '_add/in1']);
        if stride ~= 1 || currentChannels ~= config.channels
            lgraph = connectLayers(lgraph, [baseName '_shortcut_gn'], [baseName '_add/in2']);
        else
            lgraph = connectLayers(lgraph, currentInput, [baseName '_add/in2']);
        end
        
        % Final ReLU
        lgraph = addLayers(lgraph, reluLayer('Name', [baseName '_relu_final']));
        lgraph = connectLayers(lgraph, [baseName '_add'], [baseName '_relu_final']);
        
        currentInput = [baseName '_relu_final'];
        currentChannels = config.channels;
    end
end

% Final layers (without classification layer)
finalLayers = [
    globalAveragePooling2dLayer('Name', 'avg_pool')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')];

lgraph = addLayers(lgraph, finalLayers);
lgraph = connectLayers(lgraph, currentInput, 'avg_pool');

% Initialize the network
net = dlnetwork(lgraph);

% Save the network
save('seresnext101.mat', 'net');

disp('SE-ResNeXt101 network with Group Normalization created and saved successfully.');



%%
function exists = isConnectionPresent(lgraph, src, dst)
    % Convert Connections table to cell array for easier processing
    connections = lgraph.Connections;
    exists = any(strcmp(connections.Source, src) & strcmp(connections.Destination, dst));
end
