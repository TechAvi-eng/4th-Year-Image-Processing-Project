classdef BlockMatch< matlab.system.SFunSystem
%BlockMatch Block Match

 
%   Copyright 2009-2013 The MathWorks, Inc.

    methods
        function out=BlockMatch
        end

        function out=setPortDataTypeConnections(~) %#ok<STOUT>
        end

    end
    properties
        AccumulatorDataType;

        BlockSize;

        CustomAccumulatorDataType;

        CustomOutputDataType;

        CustomProductDataType;

        MatchCriteria;

        MaximumDisplacement;

        OutputDataType;

        OutputValue;

        OverflowAction;

        Overlap;

        ProductDataType;

        ReferenceFrameSource;

        RoundingMethod;

        SearchMethod;

    end
end
