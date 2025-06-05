function [net, net1_4, net5] = SEResNeXt101Backbone2(inputSize, cardinality, baseWidth)
% Build a SEResNeXt101-like backbone, then split it into:
%   - net    : full network (stages 1–5)
%   - net1_4 : stages 1–4 (removes stage 5)
%   - net5   : stage 5 only, with its own input layer [33 44 1024].
%
% Usage:
%   [net, net1_4, net5] = SEResNeXt101Backbone2([504 720 1], 32, 8);

% ------------------------------------------------
% 1) BUILD THE ENTIRE BACKBONE (STAGES 1 THROUGH 5)
% ------------------------------------------------

lgraph = layerGraph();

% ---- Stem (initial layers) ----
stemLayers = [
    imageInputLayer(inputSize, 'Name', 'Input_data')
    convolution2dLayer(7, 64, 'Stride', 2, 'Padding', 3, 'Name', 'conv1')
    groupNormalizationLayer(min(32, 64/8), 'Name', 'gn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(3, 'Stride', 2, 'Padding', 1, 'Name', 'maxpool1')];

lgraph = addLayers(lgraph, stemLayers);

% ---- Stage Configurations ----
configs = {
    struct('channels', 256,  'blocks', 3,  'stride', 1 )  % Stage 2
    struct('channels', 512,  'blocks', 4,  'stride', 2 )  % Stage 3
    struct('channels', 1024, 'blocks', 23, 'stride', 2 )  % Stage 4
    struct('channels', 2048, 'blocks', 3,  'stride', 2 )  % Stage 5
};

currentInput    = 'maxpool1'; % output of the stem
currentChannels = 64;         % after 'conv1'

% ---- Build each stage ----
for stageIdx = 1:numel(configs)
    cfg = configs{stageIdx};
    for blockIdx = 1:cfg.blocks
        stride = cfg.stride;
        if blockIdx > 1
            stride = 1;  % only the 1st block in each stage has stride
        end
        
        baseName  = sprintf('stage%d_block%d', stageIdx+1, blockIdx);
        width     = cardinality * baseWidth;
        numGroups = min(32, width/8);  % for GroupNormalization

        % ---- Main path (3 conv layers) ----
        mainPath = [
            convolution2dLayer(1, width, 'Name', [baseName '_conv1'], 'Padding','same')
            groupNormalizationLayer(numGroups, 'Name', [baseName '_gn1'])
            reluLayer('Name', [baseName '_relu1'])
            
            groupedConvolution2dLayer(3, width, cardinality, 'Stride', stride, ...
                'Name', [baseName '_conv2'], 'Padding','same')
            groupNormalizationLayer(numGroups, 'Name', [baseName '_gn2'])
            reluLayer('Name', [baseName '_relu2'])
            
            convolution2dLayer(1, cfg.channels, 'Name', [baseName '_conv3'], 'Padding','same')
            groupNormalizationLayer(numGroups, 'Name', [baseName '_gn3'])
        ];
        lgraph = addLayers(lgraph, mainPath);
        
        % ---- SE path (Global Pool -> 1x1 reduce -> ReLU -> 1x1 expand -> Sigmoid) ----
        sePath = [
            globalAveragePooling2dLayer('Name', [baseName '_gap'])
            convolution2dLayer(1, round(cfg.channels/16), 'Name', [baseName '_conv1x1_1'])
            reluLayer('Name', [baseName '_relu_se'])
            convolution2dLayer(1, cfg.channels, 'Name', [baseName '_conv1x1_2'])
            sigmoidLayer('Name', [baseName '_sigmoid'])
        ];
        lgraph = addLayers(lgraph, sePath);
        
        % ---- Connect main branch input ----
        lgraph = connectIfNotExist(lgraph, currentInput, [baseName '_conv1']);
        
        % ---- If needed, add a shortcut (1x1 conv) for mismatched dimension/stride ----
        if (stride ~= 1) || (currentChannels ~= cfg.channels)
            shortcutPath = [
                convolution2dLayer(1, cfg.channels, 'Stride', stride, ...
                    'Name', [baseName '_shortcut'])
                groupNormalizationLayer(numGroups, 'Name', [baseName '_shortcut_gn'])
            ];
            lgraph = addLayers(lgraph, shortcutPath);
            
            lgraph = connectIfNotExist(lgraph, currentInput, [baseName '_shortcut']);
        end
        
        % ---- Multiplication + Addition for SE ----
        multiplyName = [baseName '_multiply'];
        addName      = [baseName '_add'];
        
        lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', multiplyName));
        lgraph = addLayers(lgraph, additionLayer(2, 'Name', addName));
        
        % SE path -> multiply
        lgraph = connectIfNotExist(lgraph, [baseName '_gn3'],      [baseName '_gap']);
        lgraph = connectIfNotExist(lgraph, [baseName '_gap'],      [baseName '_conv1x1_1']);
        lgraph = connectIfNotExist(lgraph, [baseName '_conv1x1_1'], [baseName '_relu_se']);
        lgraph = connectIfNotExist(lgraph, [baseName '_relu_se'],   [baseName '_conv1x1_2']);
        lgraph = connectIfNotExist(lgraph, [baseName '_conv1x1_2'], [baseName '_sigmoid']);
        lgraph = connectIfNotExist(lgraph, [baseName '_sigmoid'],   [multiplyName '/in2']);
        
        % Main branch -> multiply
        lgraph = connectIfNotExist(lgraph, [baseName '_gn3'], [multiplyName '/in1']);
        
        % Then multiply -> add, plus shortcut or input
        if (stride ~= 1) || (currentChannels ~= cfg.channels)
            lgraph = connectIfNotExist(lgraph, [baseName '_shortcut_gn'], [addName '/in2']);
        else
            lgraph = connectIfNotExist(lgraph, currentInput, [addName '/in2']);
        end
        lgraph = connectIfNotExist(lgraph, multiplyName, [addName '/in1']);
        
        % ---- Final ReLU after addition ----
        finalReLU = reluLayer('Name', [baseName '_relu_final']);
        lgraph = addLayers(lgraph, finalReLU);
        lgraph = connectIfNotExist(lgraph, addName, [baseName '_relu_final']);
        
        % Update for next block
        currentInput    = [baseName '_relu_final'];
        currentChannels = cfg.channels;
    end
end

% ---- Build the FULL network from lgraph ----
net = dlnetwork(lgraph);

% -------------------------------------------------
% 2) EXTRACT net1_4: remove all stage 5 layers
% -------------------------------------------------
allLayerNames    = {lgraph.Layers.Name};
isStage5Layer    = startsWith(allLayerNames, 'stage5_');  % layers that begin with 'stage5_'
stage5LayerNames = allLayerNames(isStage5Layer);

lgraph1_4 = removeLayers(lgraph, stage5LayerNames);
net1_4    = dlnetwork(lgraph1_4);

% -------------------------------------------------
% 3) EXTRACT net5: keep only stage 5
% -------------------------------------------------
lgraph5 = removeLayers(lgraph, allLayerNames(~isStage5Layer));

% Remove old "Input_data" (if it still exists in stage5 subgraph)
if any(strcmp({lgraph5.Layers.Name}, 'Input_data'))
    lgraph5 = removeLayers(lgraph5, 'Input_data');
end

% Add a new input layer for stage5 with size [33 44 1024]
stage5Input = imageInputLayer([33 44 1024], ...
    'Name', 'stage5_input', ...
    'Normalization', 'none');
lgraph5 = addLayers(lgraph5, stage5Input);

% Connect new input to the first stage5 block's main path & (if present) shortcut
block1Main     = 'stage5_block1_conv1';
block1Shortcut = 'stage5_block1_shortcut';

if any(strcmp({lgraph5.Layers.Name}, block1Main))
    lgraph5 = connectIfNotExist(lgraph5, 'stage5_input', block1Main);
end
if any(strcmp({lgraph5.Layers.Name}, block1Shortcut))
    lgraph5 = connectIfNotExist(lgraph5, 'stage5_input', block1Shortcut);
end

net5 = dlnetwork(lgraph5);

end  % END of main function


% ----------------------------------------------
% connectIfNotExist: Checks for existing connection first
% ----------------------------------------------
function lgraph = connectIfNotExist(lgraph, src, dst)
    conn = lgraph.Connections;
    alreadyExists = any(strcmp(conn.Source, src) & strcmp(conn.Destination, dst));
    if ~alreadyExists
        lgraph = connectLayers(lgraph, src, dst);
    end
end
