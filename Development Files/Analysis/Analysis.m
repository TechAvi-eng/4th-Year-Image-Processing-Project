clear all
close all
clc

function features = shape_features(masks, bbox)

    % Initialise variables
    totalArea = 0;
    totalPerimeter = 0;
    totalWidth = 0;
    totalHeight = 0;
    numCells = size(masks, 3); % Number of cells in the image (3rd dimension of masks)
    
    % Loop through each cell 
    for i = 1:numCells
        % Extract single cell mask
        cellMask = masks(:,:,i);
        
        % Compute the area (count white pixels)
        cellArea = sum(cellMask, 'all');
        totalArea = totalArea + cellArea;
        
        % Cell Perimeter
        perimeterMask = bwperim(cellMask); % Extract cell edges
        cellPerimeter = sum(perimeterMask, 'all'); % Count perimeter pixels
        totalPerimeter = totalPerimeter + cellPerimeter;
        
        % Cell Width and Height
        cellWidth = bbox(i,3);
        totalWidth = totalWidth + cellWidth;
        cellHeight = bbox(i,4);
        totalHeight = totalHeight + cellHeight;

    end
    
    % Calculate average
    averageArea = totalArea/numCells;
    averagePerimeter = totalPerimeter/numCells;
    averageWidth = totalWidth/numCells;
    averageHeight = totalHeight/numCells;

    if averageWidth > averageHeight
        major_axis = averageWidth;
    else
        major_axis = averageHeight;
    end


    % Calculates features
    roundness = (4*averageArea)/(pi*(major_axis)^2);
    aspect_ratio = averageWidth/averageHeight;

    features = [numCells, averageArea, averagePerimeter, averageWidth, averageHeight, roundness, aspect_ratio];
    

end




% Specify folder path containing mat files
parentFolder = fileparts(pwd);
matFolder = fullfile(parentFolder, 'Data Formatting Utils', 'CatDSFs');

matFiles = dir(fullfile(matFolder, '*.mat')); % Get all .mat files

randomOrder = randperm(length(matFiles)); % Generate random indices
matFiles = matFiles(randomOrder); % Shuffle file order

% Specifying the different cell names
cellNames = {'A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3'};

for k = 1:length(matFiles)
% for k = 1:20
    matFilePath = fullfile(matFolder, matFiles(k).name);
    label = matFiles(k).name;

    data = load(matFilePath);
    
    % Display which file is being processed
    fprintf('Processing file: %s\n', label);
    
    cellType = regexp(label, '_(.*?)_', 'tokens');
    cellType = cellType{1}{1};
    
    cellNum = find(strcmp(cellType, cellNames));


    underscoreIndices = strfind(label, '_'); % Find positions of underscores
    day = str2double(label(underscoreIndices(5)+1:underscoreIndices(5)+2)); % Start index
    hour = str2double(label(underscoreIndices(5)+4:underscoreIndices(5)+5));
    time = day + hour/24;

    dataExtracted(k,1) = k;
    dataExtracted(k,2) = cellNum;
    dataExtracted(k,3) = time;

    % Extract data
    bbox = data.bbox; % Access 'bbox' variable
    masks = data.masks; % Access masks
    
    features = shape_features(masks,bbox);
    
    dataExtracted(k,4:3+length(features)) = features;

end



