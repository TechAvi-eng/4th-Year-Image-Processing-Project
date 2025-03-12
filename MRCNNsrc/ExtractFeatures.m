%bugs: some proposals result in different data types, potentially related
%to empty proposals or certain shapes?
%test with mutliple images outputted from the CNN to identify what causes
%the data types to change and make it consistent for all inputs

%add the scores (require it as an input) so that it can be
%exported easily

function extractedTable = ExtractFeatures(imageData, cellMasks, boundingBoxes, pixelToAreaRatio)

arguments
    imageData % Input image
    cellMasks % Binary masks for segmented cells
    boundingBoxes % Bounding boxes for each cell
    pixelToAreaRatio (1,1) {mustBePositive}= 1; % Conversion factor from pixels to area
end

% Normalise image if it is of type uint8
if strmatch(class(imageData),'uint8')
    imageData = rescale(imageData);
end

numCells = size(cellMasks, 3); % Determine the number of detected cells
maskIms = cellMasks .* imageData(:,:,:); % Apply masks to isolate cell regions

pixelToAreaRatio=1; % micrometers/pixel ratio

% Compute the area of each cell in micrometers squared
Area = squeeze(sum(cellMasks,1:2)) * pixelToAreaRatio^2; % Total number of pixels per mask times constant
Area = table(Area);

% Pre-allocation of data structures for improved computational speed
Perimeter = zeros(numCells, 1);
BoundingBoxHeight = zeros(numCells, 1);
BoundingBoxWidth = zeros(numCells, 1);
BoundingBoxX = zeros(numCells, 1);
BoundingBoxY = zeros(numCells, 1);

for i = 1:numCells
    % Compute perimeter of the cell mask
    Perimeter(i, 1) = sum(bwperim(cellMasks(:,:,i)),"all");

    % Compute Feret diameter
    FeretDiameter(i,:) = bwferet(cellMasks(:,:,i));

    % Compute region properties: Eccentricity, Circularity, Solidity
    ShapeProps(i,:) = regionprops(cellMasks(:,:,i), "Eccentricity", "Circularity", "Solidity");
    
    % Extract bounding box features
    BoundingBoxX(i) = boundingBoxes(i,1); % X position
    BoundingBoxY(i) = boundingBoxes(i,2); % Y position
    BoundingBoxWidth(i) = boundingBoxes(i,3); % Width
    BoundingBoxHeight(i) = boundingBoxes(i,4); % Height
end

% Convert structures to tables
ShapeProps = struct2table(ShapeProps);
Perimeter = table(Perimeter);
BoundingBoxTable = table(BoundingBoxX, BoundingBoxY, BoundingBoxWidth, BoundingBoxHeight);

% Convert diameters to micrometers (but not coordinates in the image)
FeretDiameter{:,1} = FeretDiameter{:,1} * pixelToAreaRatio;

% Phase Contrast Intensity stats for each mask
[PCIstats, IntensityDistStats] = IntensityStats(maskIms);

% Compute normalised weighted centroid using bounding boxes
[NWCx, NWCy] = weightedCentroid(maskIms, boundingBoxes);
NWC = table([NWCx NWCy]);
NWC.Properties.VariableNames = "Normalised Weighted Centroid";

% Compute aspect ratio from bounding boxes
AspectRatio = boundingBoxes(:,3)./boundingBoxes(:,4);
AspectRatio = table(AspectRatio);
AspectRatio.Properties.VariableNames = "Aspect Ratio";

% Renaming IntensityDistStats variable names
IntensityDistStats.Properties.VariableNames = ["Standard Deviation", "Skewness", "Kurtosis"];

% Combine all extracted features into a single table
extractedTable = horzcat(Area, Perimeter, PCIstats, IntensityDistStats, NWC, ShapeProps, AspectRatio, FeretDiameter, BoundingBoxTable);

end
%%


function [PCIstats, IntensityDistStats] = IntensityStats(maskIms)
Intensities = reshape(maskIms, [], size(maskIms,3), 1);
for i=[1:size(maskIms,3)]
    
    ins = Intensities(:,i);
    ins = ins(ins ~= 0); %remove 0 values
    ins = sort(ins);

    Mean(i,:) = mean(ins);
    Min(i,:) = min(ins);
    Max(i,:) = max(ins);
    P5(i,:) = ins(ceil(0.05*length(ins))); %5th percentile
    P95(i,:) = ins(ceil(0.95*length(ins))); %95th percentile

    Skewness(i,:) = skewness(ins);
    Kurtosis(i,:) = kurtosis(ins);
    STDeviation(i,:) = std(ins);
    
end

    PCIstats = table(Mean, Min, Max, P5, P95);
    IntensityDistStats = table(STDeviation, Skewness, Kurtosis);

end


function [NWCx, NWCy] = weightedCentroid(maskIms, bbox);

sizeX = size(maskIms,2);
sizeY = size(maskIms,1);

[X Y] = meshgrid(1:sizeX, 1:sizeY);

%find weighted coordinate for intensity in each mask
WCx = squeeze(sum(maskIms .* X, 1:2)./sum(maskIms, 1:2)); 
WCy = squeeze((sum(maskIms .* Y, 1:2))./sum(maskIms, 1:2)); 
%use bounding box values to find position relative to cell itself, normalize by the size of the bounding box
NWCx = (WCx - bbox(:,1))./bbox(:,3);
NWCy = (WCy - bbox(:,2))./bbox(:,4);

end