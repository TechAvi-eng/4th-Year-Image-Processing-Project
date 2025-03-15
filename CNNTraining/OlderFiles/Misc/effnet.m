clear
clc
close all
% EfficientNet V2-XL Detailed Architecture Script

% Input parameters
inputSize = [480 480 3];
numClasses = 1000;  % Default ImageNet classes

% Helper function to create MBConv block layers
function blockLayers = createMBConvBlock(blockIndex, inputChannels, outputChannels, expansionRatio, stride)
    % Expansion layer
    expandedChannels = inputChannels * expansionRatio;
    blockLayers = [
        % Expansion pointwise convolution
        convolution2dLayer(1, expandedChannels, 'Name', sprintf('expand_conv_block%d', blockIndex))
        batchNormalizationLayer('Name', sprintf('expand_bn_block%d', blockIndex))
        reluLayer('Name', sprintf('expand_relu_block%d', blockIndex))
        
        % Depthwise separable convolution
        convolution2dLayer(3, expandedChannels, 'Stride', stride, 'Padding', 'same', 'Name', sprintf('dwconv_block%d', blockIndex))
        batchNormalizationLayer('Name', sprintf('dw_bn_block%d', blockIndex))
        reluLayer('Name', sprintf('dw_relu_block%d', blockIndex))
        
        % Pointwise convolution
        convolution2dLayer(1, outputChannels, 'Name', sprintf('pwconv_block%d', blockIndex))
        batchNormalizationLayer('Name', sprintf('pw_bn_block%d', blockIndex))
    ];
end

% Create layers for EfficientNet V2-XL backbone
layers = [
    % Input layer
    imageInputLayer(inputSize, 'Name', 'input_layer')
    
    % Stem convolution
    convolution2dLayer(3, 32, 'Stride', 2, 'Padding', 'same', 'Name', 'stem_conv_1')
    batchNormalizationLayer('Name', 'stem_bn_1')
    reluLayer('Name', 'stem_relu_1')
];

% Define base configuration for MBConv blocks
mbconvConfigs = [
    % inputChannels, outputChannels, expansionRatio, stride
    32,   16,  1, 1;
    16,   32,  4, 2;
    32,   32,  4, 1;
    32,   64,  4, 2;
    64,   64,  4, 1;
    64,  128,  4, 2;
    128, 128,  4, 1;
    128, 256,  4, 2;
    256, 256,  4, 1;
    256, 512,  4, 2;
    512, 512,  4, 1;
    512, 640,  4, 1;
    640, 640,  4, 1;
];

% Add MBConv blocks
for i = 1:size(mbconvConfigs, 1)
    blockLayers = createMBConvBlock(...
        i, ...                     % block index for unique naming
        mbconvConfigs(i, 1), ...   % input channels
        mbconvConfigs(i, 2), ...   % output channels
        mbconvConfigs(i, 3), ...   % expansion ratio
        mbconvConfigs(i, 4)  ...      % stride
    );
    layers = [layers; blockLayers];
end

% Additional layers for feature extraction
layers = [
    layers;
    
    % Global average pooling
    globalAveragePooling2dLayer('Name', 'global_pool_final')
    
    % Fully connected layer for feature representation
    fullyConnectedLayer(1280, 'Name', 'feature_representation_layer')
];

% Create layer graph
lgraph = layerGraph(layers);

% Create dlnetwork
net = dlnetwork(lgraph);


% Optional: Analyze network
analyzeNetwork(net)