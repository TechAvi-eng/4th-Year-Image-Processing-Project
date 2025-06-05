
function Out = ExtractFeatures(im, masks, bbox, lpp)
arguments
    im 
    masks 
    bbox 
    lpp (1,1) {mustBePositive}= 1; %pixel to area conversion
end

% todo: make into function, rename all variables to be readable as is to simplify export (nikhil, pls do :) )
if class(im) == 'uint8'
    im = rescale(im);
end

numCells = size(masks, 3); %find number of cells
maskIms = masks .* im(:,:,:); %pixels within each maskt

lpp=1; %micrometers/pixel

Area = squeeze(sum(masks,1:2)) * lpp^2; %total number of pixels per mask times constant
Area = table(Area);

for i=[1:numCells]

    Perimeter(i, 1) = sum(bwperim(masks(:,:,i)),"all");
    FeretDiameter(i,:) = bwferet(masks(:,:,i));

    ShapeProps(i,:) = regionprops(masks(:,:,i), "Eccentricity","Circularity","Solidity");

end
ShapeProps = struct2table(ShapeProps);
Perimeter = table(Perimeter);

FeretDiameter{:,1} = FeretDiameter{:,1} * lpp; %convert diameters to micrometers (but not coordinates in the image)


% Phase Contrast Intensity stats for each mask
[PCIstats, IntensityDistStats] = IntensityStats(maskIms);

%normalized weighted centroid
[NWCx, NWCy] = weightedCentroid(maskIms, bbox);
NWC = table([NWCx NWCy]);
NWC.Properties.VariableNames="NormalizedWeightedCentroid";

%find aspect ratio from bounding boxes
AspectRatio = bbox(:,3)./bbox(:,4);
AspectRatio = table(AspectRatio);


Out = horzcat(Area, Perimeter, PCIstats, IntensityDistStats, NWC, ShapeProps, AspectRatio, FeretDiameter);

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
    STDeviaton(i,:) = std(ins);
    
end

    PCIstats = table(Mean, Min, Max, P5, P95);
    IntensityDistStats = table(STDeviaton, Skewness, Kurtosis);

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