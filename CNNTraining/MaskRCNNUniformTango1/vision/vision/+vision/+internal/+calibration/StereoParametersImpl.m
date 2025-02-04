%

%#codegen
classdef StereoParametersImpl <  vision.internal.EnforceScalarHandle

%   Copyright 2014-2023 The MathWorks, Inc.

    properties(GetAccess=public, SetAccess=protected)
        % CameraParameters1 A cameraParameters object containing
        %   the parameters of camera 1.
        CameraParameters1;
        
        % CameraParameters2 A cameraParameters object containing
        %   the parameters of camera 2.
        CameraParameters2;
    end
    
    
    properties(Dependent, SetAccess=protected)
        % FundamentalMatrix The fundamental matrix relating the two
        %    cameras. If P1 is a point in image 1 in pixels, and P2
        %    is the corresponding point in image 2, then the
        %    following equality must be true: 
        %    [P2 1] * FundamentalMatrix * [P1 1]' = 0
        FundamentalMatrix;
        
        % EssentialMatrix The essential matrix relating the two cameras. If
        %    P1 is a point in image 1 in world units, and P2
        %    is the corresponding point in image 2, then the
        %    following equality must be true: 
        %    [P2 1] * EssentialMatrix * [P1 1]' = 0
        EssentialMatrix;

        % PoseCamera2 A rigidtform3d object representing the transformation
        %    of camera 2 relative to camera 1.
        PoseCamera2;

        % MeanReprojectionError Average Euclidean distance in pixels between
        %   reprojected points and detected points over both cameras over
        %   all images over all patterns.
        MeanReprojectionError;
        
        % NumPatterns The number of calibration patterns which were used to
        %   estimate the extrinsics of the two cameras.
        NumPatterns;
        
        % WorldPoints An M-by-2 array of [x,y] world coordinates of
        %   keypoints on the calibration pattern, where M is the number of
        %   keypoints in the pattern.
        %
        %   Default: []
        WorldPoints;
        
        % WorldUnits A string describing the units, in which the
        %   WorldPoints are specified.
        %
        %   Default: 'mm'
        WorldUnits;
    end

    properties (GetAccess=public, SetAccess=protected, Hidden)
        % RotationOfCamera2 A 3-by-3 matrix representing the rotation of
        %    camera 2 relative to camera 1.
        RotationOfCamera2;
        
        % TranslationOfCamera2 A 3-element vector representing the translation
        %    of camera 2 relative to camera 1.
        TranslationOfCamera2;
    end
    
    properties(Access=protected)
        LeftSerializedLength;
        RectifyMap1;
        RectifyMap2;
        RectificationParams;
        Version = ver('vision');
    end
    
    methods
        %------------------------------------------------------------------
        % Constructor
        %------------------------------------------------------------------
        function this = StereoParametersImpl(varargin)
            this.RectificationParams = vision.internal.calibration.RectificationParameters;            
            this.RectifyMap1 = vision.internal.calibration.ImageTransformer;
            this.RectifyMap2 = vision.internal.calibration.ImageTransformer;
            
            if nargin == 1
                paramStruct = varargin{1};  
                fromStruct(this, paramStruct);
            else
                narginchk(3,4)
                camParams1 = varargin{1};
                camParams2 = varargin{2};
                [this.CameraParameters1, this.CameraParameters2] = ...
                    validateCameraParameters(camParams1, camParams2);
                
                if nargin == 4
                    R = varargin{3};
                    t = varargin{4};
                    validateRotationTranslation(R, t);
                else
                    tform = varargin{3};
                    validateattributes(tform, {'rigidtform3d', 'rigid3d'}, {'scalar'}, ...
                        'stereoParameters', 'tformOfCamera2');

                    R = tform.Rotation;
                    t = tform.Translation;
                end
                
                this.RotationOfCamera2 = R;
                this.TranslationOfCamera2 = t(:)';                
            end
        end

        %------------------------------------------------------------------
        function E = get.EssentialMatrix(this)
            t = this.TranslationOfCamera2;
            tx = [0 -t(3) t(2); t(3) 0 -t(1); -t(2) t(1) 0];
            E = tx * this.RotationOfCamera2';
        end
        
        %------------------------------------------------------------------
        function F = get.FundamentalMatrix(this)
            F = this.CameraParameters2.K' \ this.EssentialMatrix / ...
                this.CameraParameters1.K;
        end

        %------------------------------------------------------------------
        function poseCamera2 = get.PoseCamera2(this)
            poseCamera2 = rigidtform3d(this.RotationOfCamera2', this.TranslationOfCamera2);
        end
        
        %------------------------------------------------------------------
        function meanError = get.MeanReprojectionError(this)
            meanError = mean([this.CameraParameters1.MeanReprojectionError, ...
                this.CameraParameters2.MeanReprojectionError]);
        end
        
        %------------------------------------------------------------------
        function numPatterns = get.NumPatterns(this)
            numPatterns = this.CameraParameters1.NumPatterns;
        end
        
        %------------------------------------------------------------------
        function worldPoints = get.WorldPoints(this)
            worldPoints = this.CameraParameters1.WorldPoints;
        end
        
        %------------------------------------------------------------------
        function worldUnits = get.WorldUnits(this)
            worldUnits = this.CameraParameters1.WorldUnits;
        end
        
        %------------------------------------------------------------------
        function paramStruct = toStruct(this)
        % toStruct Convert a stereoParameters object into a struct.
        %   paramStruct = toStruct(stereoParams) returns a struct
        %   containing the stereo parameters, which can be used to
        %   create an identical stereoParameters object. 
        %
        %   This method is useful for C code generation. You can call
        %   toStruct, and then pass the resulting structure into the generated
        %   code, which re-creates the stereoParameters object.
            paramStruct.CameraParameters1 = toStruct(this.CameraParameters1);
            paramStruct.CameraParameters2 = toStruct(this.CameraParameters2);
            paramStruct.RotationOfCamera2    = this.RotationOfCamera2;
            paramStruct.TranslationOfCamera2 = this.TranslationOfCamera2;
            paramStruct.Version = this.Version;
            
            paramStruct.RectificationParams = toStruct(this.RectificationParams);
        end
    end
    
    methods(Hidden)
        %------------------------------------------------------------------
        function [image1Rectified, image2Rectified, R1, R2, camMatrix1, camMatrix2, Q] = ...
                rectifyStereoImagesImpl(this, image1, image2, interp, ...
                fillValues, outputView)
            
            imageClass = class(image1);
            imageSize = [size(image1, 1), size(image1, 2)];
            
            if ~(isa(image1,'double') || isa(image1,'single') || isa(image1,'uint8'))
                image1 = single(image1);
                image2 = single(image2);
                fillValues = single(fillValues);
            end
            
            needToUpdateAnything = ...
                needToUpdate(this.RectificationParams, imageSize, outputView) ||...
                needToUpdate(this.RectifyMap1, image1, outputView) ||...
                needToUpdate(this.RectifyMap2, image2, outputView);
            
            if needToUpdateAnything
                [H1, H2, R1, R2, camMatrix1, camMatrix2, Q, xBounds, yBounds, success] = ...
                    this.computeRectificationParameters(imageSize, outputView);
                
                if ~success
                    coder.internal.warning('vision:calibrate:switchValidViewToFullView');
                    % switch to full output view
                    [H1, H2, R1, R2, camMatrix1, camMatrix2, Q, xBounds, yBounds, success] = ...
                        this.computeRectificationParameters(imageSize, 'full');
                end
                
                if ~success
                    % there is no common rectangle area
                    coder.internal.error('vision:calibrate:invalidBounds');                        
                end
                    
                this.RectificationParams.update(imageSize, H1, H2, R1, R2, ...
                    camMatrix1, camMatrix2, Q, outputView, xBounds, yBounds);
                
                params = this.CameraParameters1;
                this.RectifyMap1.update(image1, params.K,...
                    params.RadialDistortion, params.TangentialDistortion, ...
                    outputView, this.RectificationParams.XBounds, ...
                    this.RectificationParams.YBounds, ...
                    this.RectificationParams.H1);
                
                params = this.CameraParameters2;
                this.RectifyMap2.update(image2, params.K,...
                    params.RadialDistortion, params.TangentialDistortion, ...
                    outputView, this.RectificationParams.XBounds, ...
                    this.RectificationParams.YBounds, ...
                    this.RectificationParams.H2);
            else
                R1         = this.RectificationParams.R1;
                R2         = this.RectificationParams.R2;
                camMatrix1 = this.RectificationParams.CameraMatrix1;
                camMatrix2 = this.RectificationParams.CameraMatrix2;
                Q          = this.RectificationParams.Q;
            end
            
            image1Rectified = transformImage(this.RectifyMap1, ...
                image1, interp, fillValues);
            
            image2Rectified = transformImage(this.RectifyMap2, ...
                image2, interp, fillValues);
            
            image1Rectified = cast(image1Rectified, imageClass);
            image2Rectified = cast(image2Rectified, imageClass);
        end
                
        %------------------------------------------------------------------
        % unrectify points from camera 1.
        % A wrapper for the inverse transformation to be passed to transformImage
        %------------------------------------------------------------------
        function pointsOut = unrectifyPoints1(this, pointsIn)
            % apply inverse projective transformation
            pointsOut = this.RectificationParams.H1.transformPointsInverse(pointsIn);
            % apply distortion
            pointsOut = this.CameraParameters1.distortPoints(pointsOut);
        end
        
        %------------------------------------------------------------------
        % unrectify points from camera 2.
        % A wrapper for the inverse transformation to be passed to transformImage
        %------------------------------------------------------------------
        function pointsOut = unrectifyPoints2(this, pointsIn)
            % apply inverse projective transformation
            pointsOut = this.RectificationParams.H2.transformPointsInverse(pointsIn);
            % apply distortion
            pointsOut = this.CameraParameters2.distortPoints(pointsOut);
        end
        
        %------------------------------------------------------------------
        function [Hleft, Hright, R1, R2, camMatrix1, camMatrix2, Q, xBounds, yBounds, success] = ...
                computeRectificationParameters(this, imageSize, outputView)
            
            % Make the two image planes coplanar, by rotating each half way
            [Rl, Rr] = computeHalfRotations(this);
            
            % rotate the translation vector
            t = Rr * this.TranslationOfCamera2';
            
            % Row align the image planes, by rotating both of them such
            % that the translation vector coincides with the X-axis.
            RrowAlign = computeRowAlignmentRotation(t);
            
            % combine rotation matrices
            R1 = RrowAlign * Rl;
            R2 = RrowAlign * Rr;
            
            Kl = this.CameraParameters1.K;
            Kr = this.CameraParameters2.K;
            K_new = computeNewIntrinsics(this); % in [fx 0 cx; 0 fy cy; 0 0 1] format
            
            Hleft  = projective2d((K_new * R1 / Kl)');
            Hright = projective2d((K_new * R2 / Kr)');
            
            % apply row alignment to translation
            t = RrowAlign * t;
            
            [xBounds, yBounds, success] = computeOutputBounds(this, ...
                imageSize, Hleft, Hright, outputView);
            
            K_new(1:2, 3) = K_new(1:2, 3) - [xBounds(1); yBounds(1)];
            camMatrix1 = [eye(3); zeros(1, 3)]  * K_new';
            camMatrix2 = [eye(3); [t(1), 0, 0]] * K_new';

            % [x, y, disparity, 1] * Q = [X, Y, Z, 1] * w
            cy = K_new(2,3);
            cx = K_new(1,3);
            f_new = K_new(2,2);
            Q = [1, 0,   0,       -cx;
                 0, 1,   0,       -cy;
                 0, 0,   0,       f_new;
                 0, 0, -1/t(1),   0]';
            
        end
        
        %------------------------------------------------------------------
        function points3D = reconstructSceneImpl(this, disparityMap)
            % check if rectifyStereoImages has been called previously with
            % the correct size images.
            coder.internal.errorIf(~this.RectificationParams.Initialized, ...
                'vision:calibrate:callRectifyFirst', 'stereoParams');
            
            disparityMapInput = 'disparityMap';
            coder.internal.errorIf(~isequal(size(disparityMap), ...
                    this.RectificationParams.RectifiedImageSize), ...
                    'vision:calibrate:disparitySizeMismatch', ...
                    disparityMapInput,...
                    this.RectificationParams.RectifiedImageSize(1), ...
                    this.RectificationParams.RectifiedImageSize(2), ...
                    disparityMapInput);
                
            % calculate the 3D locations
            % output is of the same class as disparity map
            Q = cast(this.RectificationParams.Q, 'like', disparityMap);
            
            if isempty(coder.target)
                points3D = visionReconstructScene(disparityMap, Q);
            else
                points3D = vision.internal.calibration.reconstructFromDisparity(disparityMap, Q);
            end
        end
    end
    
    methods(Access=private)
        %------------------------------------------------------------------
        function fromStruct(this, paramStruct)
            validateParamStruct(paramStruct);
            
            camParams1 = cameraParameters(paramStruct.CameraParameters1);
            camParams2 = cameraParameters(paramStruct.CameraParameters2);
            
            R = paramStruct.RotationOfCamera2;
            t = paramStruct.TranslationOfCamera2;
            
            if paramStruct.RectificationParams.Initialized
                if isfield(paramStruct.RectificationParams, 'R1')
                    R1         = paramStruct.RectificationParams.R1;
                    R2         = paramStruct.RectificationParams.R2;
                    camMatrix1 = paramStruct.RectificationParams.CameraMatrix1;
                    camMatrix2 = paramStruct.RectificationParams.CameraMatrix2;
                else
                    % Handle pre-R2022a version, which did not have the
                    % rectification parameters in RectificationParams
                    R1         = eye(3);
                    R2         = R1;
                    camMatrix1 = [eye(3); zeros(1, 3)];
                    camMatrix2 = camMatrix1;
                end

                this.RectificationParams.update(...
                    paramStruct.RectificationParams.OriginalImageSize,...
                    projective2d(paramStruct.RectificationParams.H1), ...
                    projective2d(paramStruct.RectificationParams.H2), ...
                    R1, R2, camMatrix1, camMatrix2, ...
                    paramStruct.RectificationParams.Q, ...
                    paramStruct.RectificationParams.OutputView,...
                    paramStruct.RectificationParams.XBounds,...
                    paramStruct.RectificationParams.YBounds);
            end
            
            [this.CameraParameters1, this.CameraParameters2] = ...
                validateCameraParameters(camParams1, camParams2);
            validateRotationTranslation(R, t);
            this.RotationOfCamera2 = R;
            this.TranslationOfCamera2 = t(:)';
        end
        
        %------------------------------------------------------------------
        function [Rl, Rr] = computeHalfRotations(this)
            r = vision.internal.calibration.rodriguesMatrixToVector(this.RotationOfCamera2');
            
            % right half-rotation
            Rr = vision.internal.calibration.rodriguesVectorToMatrix(r / -2);
            
            % left half-rotation
            Rl = Rr';
        end
        
        %------------------------------------------------------------------
        function K_new = computeNewIntrinsics(this)
            % initialize new camera intrinsics
            Kl = this.CameraParameters1.K;
            Kr = this.CameraParameters2.K;
            
            K_new=Kl;
            
            % find new focal length
            f_new = min([Kr(1,1),Kl(1,1)]);
            
            % set new focal lengths
            K_new(1,1)=f_new; K_new(2,2)=f_new;
            
            % find new y center
            cy_new = (Kr(2,3)+Kl(2,3)) / 2;
            
            % set new y center
            K_new(2,3)= cy_new;
            
            % set the skew to 0
            K_new(1,2) = 0;
        end
        
        %------------------------------------------------------------------
        %
        %------------------------------------------------------------------
        function [xBounds, yBounds, success] = computeOutputBounds(this, ...
                        imageSize, Hleft, Hright, outputView)
            
            % find the bounds of the undistorted images
            [xBoundsUndistort1, yBoundsUndistort1] = ...
                computeUndistortBounds(this.CameraParameters1, ...
                imageSize, outputView);
            
            undistortBounds1 = getUndistortCorners(xBoundsUndistort1, yBoundsUndistort1);
            
            [xBoundsUndistort2, yBoundsUndistort2] = ...
                computeUndistortBounds(this.CameraParameters2, ...
                imageSize, outputView);
            undistortBounds2 = getUndistortCorners(xBoundsUndistort2, yBoundsUndistort2);                        
            
            % apply the projective transformation
            outBounds1 =  Hleft.transformPointsForward(undistortBounds1);
            outBounds2 = Hright.transformPointsForward(undistortBounds2);
            
            if strcmp(outputView, 'full')
                [xBounds, yBounds, success] = computeOutputBoundsFull( ...
                    outBounds1, outBounds2);
            else % valid
                [xBounds, yBounds, success] = computeOutputBoundsValid(...
                    outBounds1, outBounds2);
            end            
        end
      end
end

%--------------------------------------------------------------
function [camParamsOut1, camParamsOut2] = validateCameraParameters(camParamsIn1, camParamsIn2)
if isa(camParamsIn1, 'cameraIntrinsics')
    validateattributes(camParamsIn1, {'cameraIntrinsics'},...
        {'scalar'}, 'stereoParameters', 'cameraParameters1');
    validateattributes(camParamsIn2, {'cameraIntrinsics'},...
        {'scalar'}, 'stereoParameters', 'cameraParameters1');
    camParamsOut1 = constructCameraParameters(camParamsIn1);
    camParamsOut2 = constructCameraParameters(camParamsIn2);
else
    validateattributes(camParamsIn1, {'cameraParameters'}, {'scalar'}, ...
        'stereoParameters', 'cameraParameters1');
    validateattributes(camParamsIn2, {'cameraParameters'}, {'scalar'}, ...
        'stereoParameters', 'cameraParameters2');

    coder.internal.errorIf(camParamsIn1.NumPatterns ~= camParamsIn2.NumPatterns,...
        'vision:calibrate:sameNumPatterns');

    % Check that the world units are the same.
    coder.internal.errorIf(~strcmpi(camParamsIn1.WorldUnits, camParamsIn2.WorldUnits),...
        'vision:calibrate:sameWorldUnits');
    camParamsOut1 = camParamsIn1;
    camParamsOut2 = camParamsIn2;
end

%------------------------------------------------------------------
function camParams = constructCameraParameters(in)
    camParams = cameraParameters('K',in.K,'RadialDistortion',in.RadialDistortion,...
        'TangentialDistortion',in.TangentialDistortion,'ImageSize',in.ImageSize);
end  
end
%--------------------------------------------------------------
function validateRotationTranslation(R, t)

validateattributes(R, {'double', 'single'}, ...
    {'nonempty', 'finite', 'real', 'nonsparse', '2d', ...
    'nrows', 3, 'ncols', 3}, 'stereoParameters', 'rotationOfCamera2'); 

validateattributes(t, {'double', 'single'}, ...
    {'nonempty', 'finite', 'real', 'nonsparse', 'vector', ...
    'numel', 3}, 'stereoParameters', 'translationOfCamera2');
end

%--------------------------------------------------------------
function validateParamStruct(paramStruct)
validateattributes(paramStruct, {'struct'}, {'scalar'}, ...
    'stereoParameters', 'paramStruct'); 

coder.internal.errorIf(~isfield(paramStruct, 'RectificationParams'), ...
    'vision:calibrate:missingRectificationParams');

if paramStruct.RectificationParams.Initialized
    validateRectificationParams(paramStruct.RectificationParams);
end
end

%--------------------------------------------------------------
function validateRectificationParams(rectificationParams)

% H1
coder.internal.errorIf(~isfield(rectificationParams, 'H1'), ...
    'vision:calibrate:missingRectificationParamsField', 'H1');
validateattributes(rectificationParams.H1, {'single', 'double'}, ...
    {'2d', 'real', 'nonsparse', 'finite', 'size', [3,3]}, ...
    'stereoParameters', 'paramStruct.RectificationParams.H1'); 

% H2
coder.internal.errorIf(~isfield(rectificationParams, 'H2'), ...
    'vision:calibrate:missingRectificationParamsField', 'H2');
validateattributes(rectificationParams.H2, {'single', 'double'}, ...
    {'2d', 'real', 'nonsparse', 'finite', 'size', [3,3]}, ...
    'stereoParameters', 'paramStruct.RectificationParams.H2'); 

% Q
coder.internal.errorIf(~isfield(rectificationParams, 'Q'), ...
    'vision:calibrate:missingRectificationParamsField', 'Q');
validateattributes(rectificationParams.Q, {'single', 'double'}, ...
    {'2d', 'real', 'nonsparse', 'finite', 'size', [4,4]}, ...
    'stereoParameters', 'paramStruct.RectificationParams.Q'); 

% XBounds
coder.internal.errorIf(~isfield(rectificationParams, 'XBounds'), ...
    'vision:calibrate:missingRectificationParamsField', 'XBounds');
validateattributes(rectificationParams.XBounds, {'single', 'double'}, ...
    {'vector', 'real', 'nonsparse', 'finite', 'size', [1, 2]}, ...
    'stereoParameters', 'paramStruct.RectificationParams.XBounds'); 

% YBounds
coder.internal.errorIf(~isfield(rectificationParams, 'YBounds'), ...
    'vision:calibrate:missingRectificationParamsField', 'YBounds');
validateattributes(rectificationParams.YBounds, {'single', 'double'}, ...
    {'vector', 'real', 'nonsparse', 'finite', 'size', [1, 2]}, ...
    'stereoParameters', 'paramStruct.RectificationParams.YBounds'); 

% OriginalImageSize
coder.internal.errorIf(~isfield(rectificationParams, 'OriginalImageSize'), ...
    'vision:calibrate:missingRectificationParamsField', 'OriginalImageSize');
validateattributes(rectificationParams.OriginalImageSize, {'single', 'double'}, ...
    {'vector', 'real', 'nonsparse', 'finite', 'size', [1, 2], 'positive', 'integer'}, ...
    'stereoParameters', 'paramStruct.RectificationParams.OriginalImageSize'); 

% OriginalImageSize
coder.internal.errorIf(~isfield(rectificationParams, 'OutputView'), ...
    'vision:calibrate:missingRectificationParamsField', 'OutputView');

validatestring(rectificationParams.OutputView, {'full', 'valid'}, ...
    'stereoParameters', 'paramStruct.RectificationParams.OutputView'); 
end

%--------------------------------------------------------------
function undistortBounds = getUndistortCorners(xBounds, yBounds)
undistortBounds = [xBounds(1), yBounds(1);
    xBounds(2), yBounds(1);
    xBounds(2), yBounds(2);
    xBounds(1), yBounds(2);];
end

%--------------------------------------------------------------
function [xBounds, yBounds, isValid] = computeOutputBoundsFull(...
    outBounds1, outBounds2)

minXY = min(outBounds1);
maxXY = max(outBounds1);
outBounds1 = [minXY; maxXY];

minXY = min(outBounds2);
maxXY = max(outBounds2);
outBounds2 = [minXY; maxXY];

minXY = round(min([outBounds1(1,:); outBounds2(1,:)]));
maxXY = round(max([outBounds1(2,:); outBounds2(2,:)]));
xBounds = [minXY(1), maxXY(1)];
yBounds = [minXY(2), maxXY(2)];
if minXY(1) >= maxXY(1) || minXY(2) >= maxXY(2)
    isValid = false;
else
    isValid = true;
end
end

%--------------------------------------------------------------
function [xBounds, yBounds, isValid] = computeOutputBoundsValid(...
    outBounds1, outBounds2)

% Compute the common rectangular area of the transformed images
outPts = [outBounds1; outBounds2];
xSort   = sort(outPts(:,1));
ySort   = sort(outPts(:,2));
xBounds = zeros(1, 2, 'like', outBounds1);
yBounds = zeros(1, 2, 'like', outBounds2);

outBounds1 = round(outBounds1);
outBounds2 = round(outBounds2);
% Detect if there is a common rectangle area that is large enough
xmin1 = min(outBounds1(:,1));
xmax1 = max(outBounds1(:,1));
xmin2 = min(outBounds2(:,1));
xmax2 = max(outBounds2(:,1));

if (xmin1 >= xmax2) || (xmax1 <= xmin2) % no overlap
    isValid = false;
else
    xBounds(1) = round(xSort(4));
    xBounds(2) = round(xSort(5));
    yBounds(1) = round(ySort(4));
    yBounds(2) = round(ySort(5));
    if xBounds(2)-xBounds(1) < 0.4 * min(xmax1-xmin1, xmax2-xmin2) % not big enough
        isValid = false;
    else
        isValid = true;
    end
end
end

%--------------------------------------------------------------------------
function RrowAlign = computeRowAlignmentRotation(t)

xUnitVector = [1;0;0];
if dot(xUnitVector, t) < 0
    xUnitVector = -xUnitVector;
end

% find the axis of rotation
rotationAxis = cross(t,xUnitVector);

if norm(rotationAxis) == 0 % no rotation
    RrowAlign = eye(3);
else
    rotationAxis = rotationAxis / norm(rotationAxis);
    
    % find the angle of rotation
    angle = acos(abs(dot(t,xUnitVector))/(norm(t)*norm(xUnitVector)));
    
    rotationAxis = angle * rotationAxis;
    
    % convert the rotation vector into a rotation matrix
    RrowAlign = vision.internal.calibration.rodriguesVectorToMatrix(rotationAxis);
end
end