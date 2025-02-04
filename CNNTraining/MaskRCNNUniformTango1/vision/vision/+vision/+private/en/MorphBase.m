classdef MorphBase< matlab.system.SFunSystem
    methods
        function out=MorphBase
        end

        function out=isInactivePropertyImpl(~) %#ok<STOUT>
        end

        function out=setPortDataTypeConnections(~) %#ok<STOUT>
        end

    end
    properties
        %Subclass must define this property
        Neighborhood;

        %NeighborhoodSource Source of neighborhood values
        %   Specify how to enter neighborhood or structuring element values as
        %   one of [{'Property'} | 'Input port']. If set to 'Property', use the
        %   Neighborhood property to specify the neighborhood or structuring
        %   element values. Otherwise, specify the neighborhood using an input
        %   to the step method. Note that structuring elements can only be
        %   specified using Neighborhood property and they cannot be used as
        %   input to the step method.
        NeighborhoodSource;

    end
end

 
%   Copyright 2004-2010 The MathWorks, Inc.

