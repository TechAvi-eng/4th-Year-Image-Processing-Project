classdef DebugLayer < nnet.layer.Layer
    methods
        function Z = predict(layer, X)
            disp('Shape before Fully Connected Layer:');
            disp(size(X));
            keyboard;  % Pause execution
            Z = X;     % Pass the input forward unchanged
        end
    end
end