function Iroi = cropImage(I, roi, varargin)
% Crops out and returns the roi from I. The roi should already be validated
% using vision.internal.detector.checkROI. I can be 2D, 3D, or 4D array.

%   Copyright 2013-2020 The MathWorks, Inc.

%#codegen
if isempty(roi)
    [~,~,N] = size(I);
    Iroi    = zeros(0,0,N,'like',I);
else
    if isempty(varargin)
        is3D = false;
    else
        is3D = coder.const(varargin{1});
    end
    
    if is3D
        
        % roi is a 1-by-6 vector
        if ~isempty(coder.target)
            % use assert to define upper bound of Iroi
            assert(roi(4) <= size(I,2));
            assert(roi(5) <= size(I,1));
            assert(roi(6) <= size(I,3));
            assert(roi(1) <= size(I,2));
            assert(roi(2) <= size(I,1));
            assert(roi(3) <= size(I,3));
        end
        
        % round and cast roi to int32 to avoid saturation of smaller integer types.
        roi = vision.internal.detector.roundAndCastToInt32(roi);
        
        % explicitly define ranges to allow unbounded roi
        r1 = roi(2);
        r2 = roi(5) + roi(2) - 1;
        
        c1 = roi(1);
        c2 = roi(4) + roi(1) - 1;
        
        d1 = roi(3);
        d2 = roi(6) + roi(3) - 1;
        
        Iroi = I(r1:r2, c1:c2, d1:d2, :, :);        
        
    else
        
        if ~isempty(coder.target)
            % use assert to define upper bound of Iroi
            assert(roi(3) <= size(I,2));
            assert(roi(4) <= size(I,1));
            assert(roi(1) <= size(I,2));
            assert(roi(2) <= size(I,1));
        end
        
        % round and cast roi to int32 to avoid saturation of smaller integer types.
        roi = vision.internal.detector.roundAndCastToInt32(roi);
        
        % explicitly define ranges to allow unbounded roi
        r1 = roi(2);
        r2 = roi(4) + roi(2) - 1;
        
        c1 = roi(1);
        c2 = roi(3) + roi(1) - 1;
        
        Iroi = I(r1:r2, c1:c2, :, :);
    end
end

end
