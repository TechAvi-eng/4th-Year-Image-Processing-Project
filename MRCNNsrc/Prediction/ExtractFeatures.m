function extractedTable = ExtractFeatures(image, masks, boundingBoxes, scores, pixeltoLengthRatio)

% Bugs
% some proposals result in different data types, potentially related
% to empty proposals or certain shapes?
% test with mutliple images outputted from the CNN to identify what causes
% the data types to change and make it consistent for all inputs

% Scores
% Functionality to store confidence scores for each cell added.
% Requires ammending once the CNN output/function input has the scores
% added.



arguments
    image % Input image
    masks % Binary masks for segmented cells
    boundingBoxes % Bounding boxes for each cell
    scores = []; % Scores input from CNN
    pixeltoLengthRatio (1,1) {mustBePositive} = 1; % Conversion factor from pixels to length
end

% Normalise image if it is of type uint8
if strmatch(class(image),'uint8')
    image = rescale(image);
end

numCells = size(masks, 3); % Determine the number of detected cells
maskIms = masks .* image(:,:,:); % Apply masks to isolate cell regions

pixeltoLengthRatio=1; % micrometers/pixel ratio

% Compute the area of each cell in micrometers squared
Area = squeeze(sum(masks,1:2)) * pixeltoLengthRatio^2; % Total number of pixels per mask times constant
Confluency = sum(Area) ./ (size(image, 1)*size(image, 2));


Area = table(Area);

Confluency = repmat(Confluency, [height(Area), 1]);
Confluency = table(Confluency);

% Pre-allocation of data structures for improved computational speed
Perimeter = zeros(numCells, 1);
BoundingBoxHeight = zeros(numCells, 1);
BoundingBoxWidth = zeros(numCells, 1);
BoundingBoxX = zeros(numCells, 1);
BoundingBoxY = zeros(numCells, 1);
scoresCells = zeros(numCells, 1);

for i = 1:numCells
    % Compute perimeter of the cell mask
    Perimeter(i, 1) = sum(bwperim(masks(:,:,i)),"all");

    % Compute Feret diameter
    FeretDiameter(i,:) = bwferet(masks(:,:,i));

    % Compute region properties: Eccentricity, Circularity, Solidity
    ShapeProps(i,:) = regionprops(masks(:,:,i), "Eccentricity", "Circularity", "Solidity");
    


end

% Convert structures to tables
ShapeProps = struct2table(ShapeProps);
Perimeter = table(Perimeter);
BoundingBoxTable = table(boundingBoxes(:,1),boundingBoxes(:,2),boundingBoxes(:,3),boundingBoxes(:,4),'VariableNames', ["Bounding Box X", "Bounding Box Y", "Bounding Box Width", "Bounding Box Height"] );
if size(scores,1)==size(boundingBoxes,1)
ScoresTable = table(scores);
else
ScoresTable = table(zeros(size(boundingBoxes,1), 1));
end

% Convert diameters to micrometers (but not coordinates in the image)
FeretDiameter{:,1} = FeretDiameter{:,1} * pixeltoLengthRatio;

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

% Renaming BoundingBoxTable variable names
%BoundingBoxTable.Properties.VariableNames = ["Bounding Box X", "Bounding Box Y", "Bounding Box Width", "Bounding Box Height"];

% Renaming ScoresTable variable names
ScoresTable.Properties.VariableNames = ["Confidence"];

% Combine all extracted features into a single table
extractedTable = horzcat(ScoresTable, Area, Perimeter, BoundingBoxTable, Confluency, AspectRatio, PCIstats, IntensityDistStats, NWC, ShapeProps, FeretDiameter);

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
    Intensity5Percentile(i,:) = ins(ceil(0.05*length(ins))); %5th percentile
    Intensity95Percentile(i,:) = ins(ceil(0.95*length(ins))); %95th percentile

    Skewness(i,:) = skewness(ins);
    Kurtosis(i,:) = kurtosis(ins);
    STDeviation(i,:) = std(ins);
    
end

    PCIstats = table(Mean, Min, Max, Intensity5Percentile, Intensity95Percentile);
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