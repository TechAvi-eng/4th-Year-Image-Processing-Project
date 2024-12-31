% rotationMatrixToVector Convert a 3-D rotation matrix into a rotation vector
%
%--------------------------------------------------------------------------
% rotationMatrixToVector is not recommended. Use rotmat2vec3d instead.
%--------------------------------------------------------------------------
%
% rotationVector = rotationMatrixToVector(rotationMatrix) computes a
% rotation vector (axis-angle representation) corresponding to a 3-D
% rotation matrix using the Rodrigues formula. rotationMatrix is a
% 3-by-3 3-D rotation matrix. rotationVector is a 3-element rotation vector
% corresponding to the rotationMatrix. The vector represents the axis
% of rotation in 3-D, and its magnitude is the rotation angle in radians.
%
% Class Support
% -------------
% rotationMatrix can be double or single. rotationVector is the same
% class as rotationMatrix.
%
% Example
% -------
% % Create a matrix representing 90-degree rotation about Z-axis
% theta = 90;
% rotationMatrix = [cosd(theta)     sind(theta)    0;
%                  -sind(theta)     cosd(theta)    0;
%                       0               0          1];
%
% % Find the equivalent rotation vector
% rotationVector = rotationMatrixToVector(rotationMatrix)
%
% See also rotvec2mat3d, cameraParameters, estrelpose, estworldpose,
%          estimateExtrinsics.

% Copyright 2015-2022 The MathWorks, Inc.

% References:
% [1] R. Hartley, A. Zisserman, "Multiple View Geometry in Computer
%     Vision," Cambridge University Press, 2003.
%
% [2] E. Trucco, A. Verri. "Introductory Techniques for 3-D Computer
%     Vision," Prentice Hall, 1998.

%#codegen

function rotationVector = rotationMatrixToVector(rotationMatrix)

validateattributes(rotationMatrix, {'single', 'double'}, ...
    {'real', 'nonsparse', 'size', [3 3]}, mfilename, 'rotationMatrix');

rotationVector = vision.internal.calibration.rodriguesMatrixToVector(rotationMatrix')';
