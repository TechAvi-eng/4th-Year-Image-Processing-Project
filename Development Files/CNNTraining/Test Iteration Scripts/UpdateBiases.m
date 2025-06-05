%function aa = quickFixReLUActivations(aa)
    % For each group normalization layer in the aawork
aa = obj.FeatureExtractionNet;

    for i = 1:height(aa.Learnables)
        layerName = aa.Learnables.Layer{i};
        paramName = aa.Learnables.Parameter{i};
        
        % Find group normalization offset parameters
        if contains(layerName, 'gn') && strcmp(paramName, 'Offset')
            % Apply a small positive shift to the offset parameter
            % This shifts the distribution to have more positive values
            offsetParam = aa.Learnables{i,3};
            offsetParam = offsetParam{1,1};
            % Calculate an appropriate shift amount (try 0.5 first)
            shiftAmount = 0.5;
            newWeights{1,1} = offsetParam + shiftAmount;
            % Apply the shift
            aa.Learnables{i,3} = newWeights;
        end
    end
obj.FeatureExtractionNet = aa;


    %end