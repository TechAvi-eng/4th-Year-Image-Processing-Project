function trainingImageLabeler(varargin)
%trainingImageLabeler
%
%   The Training Image Labeler app has been replaced by the Image Labeler
%   app. Use imageLabeler launch the app.
%
% See also imageLabeler.

% Copyright 2012-2017 The MathWorks, Inc.

if nargin > 0
    [varargin{:}] = convertStringsToChars(varargin{:});
end

warning(message('vision:imageLabeler:TILWarning'));
imageLabeler(varargin{:});



