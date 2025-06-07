function extractedTable = ExtractFeatures(image, masks, boundingBoxes, scores, pixeltoLengthRatio)
% EXTRACTFEATURES Computes comprehensive morphological and intensity features from segmented cells
%
% This function extracts a wide range of quantitative features from segmented cell images
% for subsequent analysis, classification, or quality assessment. It combines geometric
% measurements, intensity statistics, and spatial characteristics into a unified table.
%
% Feature categories extracted:
% - Morphological: Area, perimeter, shape descriptors, Feret diameter
% - Intensity: Statistical moments, percentiles, distribution properties  
% - Spatial: Centroids, aspect ratios, bounding box properties
% - Quality: Segmentation confidence scores, confluency metrics
%
% Outputs:
%   extractedTable - Comprehensive table with one row per cell containing all features

arguments
    image                                           % Input image (grayscale or RGB)
    masks                                           % Binary masks for segmented cells (H x W x N)
    boundingBoxes                                   % Bounding boxes [x y width height] for each cell
    scores = [];                                    % CNN confidence scores for each detection
    pixeltoLengthRatio (1,1) {mustBePositive} = 1; % Conversion factor: micrometers per pixel
end

%% Image Preprocessing
% Ensure image is in normalized [0,1] range for consistent intensity calculations
if strmatch(class(image), 'uint8')
    image = rescale(image);
end

%% Basic Cell Quantification
numCells = size(masks, 3);                          % Count total detected cells
if numCells == 0 %return early if no input data
    extractedTable = [];
    return
end

maskIms = masks .* image(:,:,:);                    % Apply masks to isolate individual cell regions

pixeltoLengthRatio = 1;                             % Set physical scale (micrometers/pixel)

%% Morphological Feature Extraction
% Calculate cell area in physical units (micrometers squared)
Area = squeeze(sum(masks, 1:2)) * pixeltoLengthRatio^2;

% Calculate confluency: fraction of image area occupied by cells
Confluency = sum(Area) ./ (size(image, 1)*size(image, 2));

% Convert to table format for consistency
Area = table(Area);
Confluency = repmat(Confluency, [height(Area), 1]);
Confluency = table(Confluency);

% Pre-allocate arrays for computational efficiency
Perimeter = zeros(numCells, 1);
BoundingBoxHeight = zeros(numCells, 1);
BoundingBoxWidth = zeros(numCells, 1);
BoundingBoxX = zeros(numCells, 1);
BoundingBoxY = zeros(numCells, 1);
scoresCells = zeros(numCells, 1);

%% Per-Cell Feature Computation
for i = 1:max(numCells,1);
    maskA = bwareafilt(masks(:,:,i), 1,"largest");
    if sum(maskA,"all")==0;
        maskA = zeros(size(image));
        maskA(1,1) = 1;
    end
    % Calculate perimeter using boundary detection
    Perimeter(i, 1) = sum(bwperim(maskA), "all");

    % Compute Feret diameter (maximum distance between boundary points)

    FeretDiameter(i,:) = bwferet(maskA);

    % Extract shape descriptors: how elongated, circular, and solid each cell is
    ShapeProps(i,:) = regionprops(maskA, "Eccentricity", "Circularity", "Solidity");
    
end

%% Data Structure Preparation
% Convert computed features to table format for consistent handling
ShapeProps = struct2table(ShapeProps);
Perimeter = table(Perimeter);

% Organize bounding box data with descriptive variable names
BoundingBoxTable = table(boundingBoxes(:,1), boundingBoxes(:,2), boundingBoxes(:,3), boundingBoxes(:,4), ...
    'VariableNames', ["Bounding Box X", "Bounding Box Y", "Bounding Box Width", "Bounding Box Height"]);

% Handle confidence scores with validation
if size(scores,1) == size(boundingBoxes,1)
    ScoresTable = table(scores);
else
    % Use zeros if scores don't match number of detections
    ScoresTable = table(zeros(size(boundingBoxes,1), 1));
end

% Convert Feret diameter to physical units (micrometers)
FeretDiameter{:,1} = FeretDiameter{:,1} * pixeltoLengthRatio;
FeretDiameter.Properties.VariableNames = ["Max Diameter", "Max Angle", "Max Coordinates"]

%% Intensity Feature Extraction
% Compute comprehensive intensity statistics for each segmented cell
[PCIstats, IntensityDistStats] = IntensityStats(maskIms);

%% Spatial Feature Computation
% Calculate normalized weighted centroids relative to bounding boxes
[NWCx, NWCy] = weightedCentroid(maskIms, boundingBoxes);
NWC = table([NWCx NWCy]);
NWC.Properties.VariableNames = "Normalised Weighted Centroid";

% Compute aspect ratio from bounding box dimensions
AspectRatio = boundingBoxes(:,3)./boundingBoxes(:,4);
AspectRatio = table(AspectRatio);
AspectRatio.Properties.VariableNames = "Aspect Ratio";

%% Final Table Assembly with Descriptive Names
% Assign meaningful variable names to intensity statistics
IntensityDistStats.Properties.VariableNames = ["Standard Deviation", "Skewness", "Kurtosis"];
ScoresTable.Properties.VariableNames = ["Confidence"];

CellNo = [1:numCells]';
CellNo = table(CellNo);

% Combine all feature categories into comprehensive output table
extractedTable = horzcat(CellNo, ScoresTable, Area, Perimeter, BoundingBoxTable, Confluency, ...
                        AspectRatio, PCIstats, IntensityDistStats, NWC, ShapeProps, FeretDiameter);

end

%%
function [PCIstats, IntensityDistStats] = IntensityStats(maskIms)
% INTENSITYSTATS Computes statistical descriptors of pixel intensities within each cell mask
%
% This function calculates comprehensive intensity statistics for each segmented cell,
% providing both basic descriptive statistics and distribution shape measures.
% These features help characterize cellular appearance and can indicate cell state,
% type, or imaging conditions.
%
% Features computed:
% - Central tendency: Mean
% - Range: Min, Max, 5th/95th percentiles (robust outlier handling)
% - Distribution shape: Skewness, kurtosis, standard deviation

% Reshape mask images for efficient vectorized processing
Intensities = reshape(maskIms, [], size(maskIms,3), 1);

for i = [1:size(maskIms,3)]
    % Extract non-zero intensities (pixels within current cell mask)
    ins = Intensities(:,i);
    ins = ins(ins ~= 0);        % Remove background pixels
    ins = sort(ins);            % Sort for percentile calculations
    
    if isempty(ins);
        ins = 0.00001;
    end
    
    % Basic statistical measures
    Mean(i,1) = mean(ins);
    Min(i,1) = min(ins,[],"all");
    Max(i,1) = max(ins,[],"all");
    
    % Robust percentile measures (less sensitive to outliers than min/max)
    Intensity5Percentile(i,:) = ins(ceil(0.05*length(ins)));   % 5th percentile
    Intensity95Percentile(i,:) = ins(ceil(0.95*length(ins)));  % 95th percentile
    
    % Distribution shape descriptors
    Skewness(i,:) = skewness(ins);      % Asymmetry of intensity distribution
    Kurtosis(i,:) = kurtosis(ins);      % Tail heaviness of distribution
    STDeviation(i,:) = std(ins);        % Spread of intensities
end

% Package statistics into separate tables for different feature types
PCIstats = table(Mean, Min, Max, Intensity5Percentile, Intensity95Percentile);
PCIstats.Properties.VariableNames(1)= "Mean Intensity";  
IntensityDistStats = table(STDeviation, Skewness, Kurtosis);

end

function [NWCx, NWCy] = weightedCentroid(maskIms, bbox)
% WEIGHTEDCENTROID Computes intensity-weighted centroids normalized by bounding box position
%
% This function calculates where the "center of mass" of each cell lies based on
% intensity weighting, then normalizes this position relative to the cell's bounding box.
% This provides a measure of internal intensity distribution and asymmetry within cells.
%
% The normalization makes centroids comparable across cells of different sizes and
% positions, with values in [0,1] representing relative position within each cell.
%
% Outputs:
%   NWCx, NWCy - Normalized weighted centroid coordinates [0,1] relative to bounding box

% Get image dimensions for coordinate grid generation
sizeX = size(maskIms,2);  
sizeY = size(maskIms,1);

% Create coordinate grids for weighted centroid calculation
[X, Y] = meshgrid(1:sizeX, 1:sizeY);

% Calculate intensity-weighted centroids for each cell
% Sum of (intensity × position) divided by sum of intensities
WCx = squeeze(sum(maskIms .* X, 1:2)./sum(maskIms, 1:2)); 
WCy = squeeze((sum(maskIms .* Y, 1:2))./sum(maskIms, 1:2)); 

% Normalize centroids relative to bounding box position and size
% This makes centroids comparable across different cell sizes and locations
NWCx = (WCx - bbox(:,1))./bbox(:,3);  % (centroid_x - bbox_left) / bbox_width
NWCy = (WCy - bbox(:,2))./bbox(:,4);  % (centroid_y - bbox_top) / bbox_height

end