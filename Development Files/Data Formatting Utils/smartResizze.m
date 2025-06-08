function result = smartResize(img, targetSize)
    % SMARTRESIZE Recursively resize an image or split into tiles
    %
    % Inputs:
    %   img - Input image (grayscale or color)
    %   targetSize - [height, width] desired output size
    %
    % Output:
    %   result - Either a single resized image or cell array of image tiles
    
    result = smartResizeRecursive(img, targetSize);
end

function result = smartResizeRecursive(img, targetSize)
    % Internal recursive function
    [h, w, channels] = size(img);
    targetH = targetSize(1);
    targetW = targetSize(2);
    
    % Calculate scaling factors
    scaleH = h / targetH;
    scaleW = w / targetW;
    maxScale = max(scaleH, scaleW);
    
    % Base case: image is not significantly larger than target
    if maxScale <= 1.5
        result = resizeAndPad(img, targetSize);
    else
        % Recursive case: split image and process each piece
        result = splitAndResize(img, targetSize);
    end
end

function result = resizeAndPad(img, targetSize)
    % Helper function to resize and pad a single image
    [h, w, channels] = size(img);
    targetH = targetSize(1);
    targetW = targetSize(2);
    
    % Calculate new dimensions maintaining aspect ratio
    scaleH = h / targetH;
    scaleW = w / targetW;
    
    if scaleH > scaleW
        % Height is the limiting factor (image is relatively tall)
        newH = targetH;
        newW = round(w * targetH / h);
    else
        % Width is the limiting factor (image is relatively wide)
        newW = targetW;
        newH = round(h * targetW / w);
    end
    
    % For small images, ensure we scale up to at least fit one dimension
    if max(scaleH, scaleW) < 1
        % Image is smaller than target - scale up to fill target optimally
        upscaleFactor = min(targetH / h, targetW / w);
        newH = round(h * upscaleFactor);
        newW = round(w * upscaleFactor);
    end
    
    % Resize the image
    resized = imresize(img, [newH, newW]);
    
    % Zero-pad to exact target size
    result = zeros(targetH, targetW, channels, 'like', img);
    
    % Center the resized image
    startH = max(1, round((targetH - newH) / 2) + 1);
    startW = max(1, round((targetW - newW) / 2) + 1);
    endH = min(targetH, startH + newH - 1);
    endW = min(targetW, startW + newW - 1);
    
    % Handle cases where resized image might be slightly larger than target
    cropH = min(newH, endH - startH + 1);
    cropW = min(newW, endW - startW + 1);
    
    result(startH:endH, startW:endW, :) = resized(1:cropH, 1:cropW, :);
end

function result = splitAndResize(img, targetSize)
    % Helper function to split image and recursively process each piece
    [h, w, ~] = size(img);
    targetH = targetSize(1);
    targetW = targetSize(2);
    
    % Determine optimal split strategy
    scaleH = h / targetH;
    scaleW = w / targetW;
    
    % Decide whether to split horizontally, vertically, or both
    if scaleH > scaleW * 1.2
        % Image is much taller than wide relative to target - split horizontally
        splitRows = 2;
        splitCols = 1;
    elseif scaleW > scaleH * 1.2
        % Image is much wider than tall relative to target - split vertically
        splitRows = 1;
        splitCols = 2;
    else
        % Image is roughly proportional - split both ways
        splitRows = 2;
        splitCols = 2;
    end
    
    % For very large images, increase split factor
    maxScale = max(scaleH, scaleW);
    if maxScale > 6
        extraSplits = floor(log2(maxScale / 3));
        if scaleH > scaleW
            splitRows = splitRows + extraSplits;
        else
            splitCols = splitCols + extraSplits;
        end
    end
    
    % Create 1D cell array to store results
    totalTiles = splitRows * splitCols;
    result = cell(1, totalTiles);
    
    % Split image and recursively process each tile
    tileIndex = 1;
    for r = 1:splitRows
        for c = 1:splitCols
            % Calculate tile boundaries with proper distribution
            rowSize = floor(h / splitRows);
            colSize = floor(w / splitCols);
            
            % Handle remainder pixels
            extraRows = mod(h, splitRows);
            extraCols = mod(w, splitCols);
            
            % Calculate start positions
            if r <= extraRows
                startH = (r-1) * (rowSize + 1) + 1;
                currentRowSize = rowSize + 1;
            else
                startH = extraRows * (rowSize + 1) + (r - extraRows - 1) * rowSize + 1;
                currentRowSize = rowSize;
            end
            
            if c <= extraCols
                startW = (c-1) * (colSize + 1) + 1;
                currentColSize = colSize + 1;
            else
                startW = extraCols * (colSize + 1) + (c - extraCols - 1) * colSize + 1;
                currentColSize = colSize;
            end
            
            % Calculate end positions
            endH = startH + currentRowSize - 1;
            endW = startW + currentColSize - 1;
            
            % Extract tile
            tile = img(startH:endH, startW:endW, :);
            
            % Recursively process this tile
            tileResult = smartResizeRecursive(tile, targetSize);
            
            % Flatten the result if it's a cell array (from further subdivision)
            if iscell(tileResult)
                % Add all sub-tiles to the 1D result array
                for k = 1:length(tileResult)
                    if tileIndex <= totalTiles
                        result{tileIndex} = tileResult{k};
                        tileIndex = tileIndex + 1;
                    else
                        % Need to expand the result array
                        result{end+1} = tileResult{k};
                        tileIndex = tileIndex + 1;
                    end
                end
            else
                % Single tile result
                result{tileIndex} = tileResult;
                tileIndex = tileIndex + 1;
            end
        end
    end
    
    % Trim any unused cells
    result = result(1:tileIndex-1);
end