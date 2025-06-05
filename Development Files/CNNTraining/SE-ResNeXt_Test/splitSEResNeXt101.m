function [net_stages1to4, net_stage5] = splitSEResNeXt101(inputSize, cardinality, baseWidth)
    % Create the full network first
    lgraph = layerGraph();
    
    % Input layer and stage 1
    tempLayers = [
        imageInputLayer(inputSize, 'Name', 'Input_data')
        convolution2dLayer([7 7], 64, 'Stride', 2, 'Padding', 3, 'Name', 'conv1')
        groupNormalizationLayer(min(32, 64 / 8), 'Name', 'gn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(3, 'Stride', 2, 'Padding', 1, 'Name', 'maxpool1')];
    
    lgraph = addLayers(lgraph, tempLayers);
    
    % Layer configurations
    configs = {
        struct('channels', 256, 'blocks', 3, 'stride', 1),   % Stage 2
        struct('channels', 512, 'blocks', 4, 'stride', 2),   % Stage 3
        struct('channels', 1024, 'blocks', 23, 'stride', 2)  % Stage 4
    };
    
    currentInput = 'maxpool1';
    currentChannels = 64;
    
    % Add stages 2-4
    for stageIdx = 1:length(configs)
        config = configs{stageIdx};
        
        for blockIdx = 1:config.blocks
            stride = config.stride;
            if blockIdx > 1
                stride = 1;
            end
            
            baseName = sprintf('stage%d_block%d', stageIdx+1, blockIdx);
            width = cardinality * baseWidth;
            numGroups = min(32, width / 8);
            
            % Main path layers
            mainPath = [
                convolution2dLayer(1, width, 'Name', [baseName '_conv1'], 'Padding', 'same')
                groupNormalizationLayer(numGroups, 'Name', [baseName '_gn1'])
                reluLayer('Name', [baseName '_relu1'])
                
                groupedConvolution2dLayer(3, width, cardinality, 'Stride', stride, ...
                    'Name', [baseName '_conv2'], 'Padding', 'same')
                groupNormalizationLayer(numGroups, 'Name', [baseName '_gn2'])
                reluLayer('Name', [baseName '_relu2'])
                
                convolution2dLayer(1, config.channels, 'Name', [baseName '_conv3'], 'Padding', 'same')
                groupNormalizationLayer(min(32, config.channels / 8), 'Name', [baseName '_gn3'])];
            
            % SE block
            % SE block (Replaced fully connected layers with 1x1 convolutions)
            sePath = [
                globalAveragePooling2dLayer('Name', [baseName '_gap'])
                convolution2dLayer(1, round(config.channels / 16), 'Name', [baseName '_conv_fc1'], 'Padding', 'same')
                reluLayer('Name', [baseName '_relu_se'])
                convolution2dLayer(1, config.channels, 'Name', [baseName '_conv_fc2'], 'Padding', 'same')
                sigmoidLayer('Name', [baseName '_sigmoid'])];

            
            % Add layers and build connections
            lgraph = addLayers(lgraph, mainPath);
            lgraph = addLayers(lgraph, sePath);
            lgraph = connectLayers(lgraph, currentInput, [baseName '_conv1']);
            
            % Add shortcut if needed
            if stride ~= 1 || currentChannels ~= config.channels
                shortcut = [
                    convolution2dLayer(1, config.channels, 'Stride', stride, ...
                        'Name', [baseName '_shortcut'])
                    groupNormalizationLayer(min(32, config.channels / 8), 'Name', [baseName '_shortcut_gn'])];
                lgraph = addLayers(lgraph, shortcut);
                lgraph = connectLayers(lgraph, currentInput, [baseName '_shortcut']);
                shortcutOutput = [baseName '_shortcut_gn'];
            else
                shortcutOutput = currentInput;
            end
            
            % Connect SE block
            lgraph = connectLayers(lgraph, [baseName '_gn3'], [baseName '_gap']);
            lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', [baseName '_multiply']));
            lgraph = addLayers(lgraph, additionLayer(2, 'Name', [baseName '_add']));
            
            % Connect multiplication and addition layers
            lgraph = connectLayers(lgraph, [baseName '_sigmoid'], [baseName '_multiply/in2']);
            lgraph = connectLayers(lgraph, [baseName '_gn3'], [baseName '_multiply/in1']);
            lgraph = connectLayers(lgraph, [baseName '_multiply'], [baseName '_add/in1']);
            lgraph = connectLayers(lgraph, shortcutOutput, [baseName '_add/in2']);
            
            % Final ReLU
            lgraph = addLayers(lgraph, reluLayer('Name', [baseName '_relu_final']));
            lgraph = connectLayers(lgraph, [baseName '_add'], [baseName '_relu_final']);
            
            currentInput = [baseName '_relu_final'];
            currentChannels = config.channels;
        end
    end
    
    % Create stage 5 network
    lgraph_stage5 = layerGraph();
    inputLayer5 = imageInputLayer([33 44 1024], 'Name', 'stage5_input');
    lgraph_stage5 = addLayers(lgraph_stage5, inputLayer5);
    
    % Stage 5 configuration
    config = struct('channels', 2048, 'blocks', 3, 'stride', 2);
    currentInput = 'stage5_input';
    currentChannels = 1024;
    
    % Build stage 5
    for blockIdx = 1:config.blocks
        stride = config.stride;
        if blockIdx > 1
            stride = 1;
        end
        
        baseName = sprintf('stage5_block%d', blockIdx);
        width = cardinality * baseWidth;
        numGroups = min(32, width / 8);
        
        % Main path layers
        mainPath = [
            convolution2dLayer(1, width, 'Name', [baseName '_conv1'], 'Padding', 'same')
            groupNormalizationLayer(numGroups, 'Name', [baseName '_gn1'])
            reluLayer('Name', [baseName '_relu1'])
            
            groupedConvolution2dLayer(3, width, cardinality, 'Stride', stride, ...
                'Name', [baseName '_conv2'], 'Padding', 'same')
            groupNormalizationLayer(numGroups, 'Name', [baseName '_gn2'])
            reluLayer('Name', [baseName '_relu2'])
            
            convolution2dLayer(1, config.channels, 'Name', [baseName '_conv3'], 'Padding', 'same')
            groupNormalizationLayer(min(32, config.channels / 8), 'Name', [baseName '_gn3'])];
        
        % SE block
        sePath = [
            globalAveragePooling2dLayer('Name', [baseName '_gap'])
            fullyConnectedLayer(round(config.channels / 16), 'Name', [baseName '_fc1'])
            reluLayer('Name', [baseName '_relu_se'])
            fullyConnectedLayer(config.channels, 'Name', [baseName '_fc2'])
            sigmoidLayer('Name', [baseName '_sigmoid'])];
        
        % Add and connect layers
        lgraph_stage5 = addLayers(lgraph_stage5, mainPath);
        lgraph_stage5 = addLayers(lgraph_stage5, sePath);
        lgraph_stage5 = connectLayers(lgraph_stage5, currentInput, [baseName '_conv1']);
        
        % Add shortcut if needed
        if stride ~= 1 || currentChannels ~= config.channels
            shortcut = [
                convolution2dLayer(1, config.channels, 'Stride', stride, ...
                    'Name', [baseName '_shortcut'])
                groupNormalizationLayer(min(32, config.channels / 8), 'Name', [baseName '_shortcut_gn'])];
            lgraph_stage5 = addLayers(lgraph_stage5, shortcut);
            lgraph_stage5 = connectLayers(lgraph_stage5, currentInput, [baseName '_shortcut']);
            shortcutOutput = [baseName '_shortcut_gn'];
        else
            shortcutOutput = currentInput;
        end
        
        % Connect SE block
        lgraph_stage5 = connectLayers(lgraph_stage5, [baseName '_gn3'], [baseName '_gap']);
        lgraph_stage5 = addLayers(lgraph_stage5, multiplicationLayer(2, 'Name', [baseName '_multiply']));
        lgraph_stage5 = addLayers(lgraph_stage5, additionLayer(2, 'Name', [baseName '_add']));
        
        % Connect multiplication and addition layers
        lgraph_stage5 = connectLayers(lgraph_stage5, [baseName '_sigmoid'], [baseName '_multiply/in2']);
        lgraph_stage5 = connectLayers(lgraph_stage5, [baseName '_gn3'], [baseName '_multiply/in1']);
        lgraph_stage5 = connectLayers(lgraph_stage5, [baseName '_multiply'], [baseName '_add/in1']);
        lgraph_stage5 = connectLayers(lgraph_stage5, shortcutOutput, [baseName '_add/in2']);
        
        % Final ReLU
        lgraph_stage5 = addLayers(lgraph_stage5, reluLayer('Name', [baseName '_relu_final']));
        lgraph_stage5 = connectLayers(lgraph_stage5, [baseName '_add'], [baseName '_relu_final']);
        
        currentInput = [baseName '_relu_final'];
        currentChannels = config.channels;
    end
    
    % Create the final networks
    net_stages1to4 = dlnetwork(lgraph);
    net_stage5 = dlnetwork(lgraph_stage5);
end