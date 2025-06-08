function result = SmartResize(img, targetSize)
    % SMARTRESIZE Intelligently resize an image or split into tiles
    %
    % Inputs:
    %   img - Input image (grayscale or color)
    %   targetSize - [height, width] desired output size
    %
    % Output:
    %   result - Either a single resized image or cell array of image tiles
    
    % Get current image dimensions
    [h, w, channels] = size(img);
    targetH = targetSize(1);
    targetW = targetSize(2);
    
    % Calculate scaling factors
    scaleH = h / targetH;
    scaleW = w / targetW;
    maxScale = max(scaleH, scaleW);
    
    % If image is not significantly larger than target, resize directly
    if maxScale <= 1.5
        % Simple resize maintaining aspect ratio
        % Scale up to fill as much of the target as possible without exceeding it
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
        if maxScale < 1
            % Image is smaller than target - scale up to fill target optimally
            upscaleFactor = min(targetH / h, targetW / w);
            newH = round(h * upscaleFactor);
            newW = round(w * upscaleFactor);
        end
        
        % Resize the image
        resized = imresize(img, [newH, newW]);
        
        % Zero-pad to exact target size if needed
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
        
    else
        % Image is too large, split into tiles
        
        % Determine optimal grid size
        % Try different grid configurations and pick the best one
        minPaddingRatio = inf;
        bestGrid = [1, 1];
        
        % Test grid sizes from 1x2 up to reasonable limits
        maxTiles = min(16, ceil(maxScale^2)); % Reasonable upper limit
        
        for totalTiles = 2:maxTiles
            % Try different grid arrangements for this number of tiles
            for rows = 1:sqrt(totalTiles)
                cols = ceil(totalTiles / rows);
                if rows * cols <= totalTiles + 1 % Allow some flexibility
                    
                    % Calculate tile dimensions
                    tileH = ceil(h / rows);
                    tileW = ceil(w / cols);
                    
                    % Check if tiles would need reasonable padding
                    paddingH = max(0, targetH - tileH);
                    paddingW = max(0, targetW - tileW);
                    totalPadding = paddingH * targetW + paddingW * targetH;
                    paddingRatio = totalPadding / (targetH * targetW);
                    
                    % Prefer grids that minimize padding and tile size variance
                    if paddingRatio < 0.3 && paddingRatio < minPaddingRatio
                        minPaddingRatio = paddingRatio;
                        bestGrid = [rows, cols];
                    end
                end
            end
        end
        
        % If no good grid found, use simple 2x2
        if isinf(minPaddingRatio)
            bestGrid = [2, 2];
        end
        
        rows = bestGrid(1);
        cols = bestGrid(2);
        
        % Calculate actual tile dimensions
        tileH = ceil(h / rows);
        tileW = ceil(w / cols);
        
        % Create cell array to store tiles
        result = cell(rows, cols);
        
        % Split image into tiles
        for r = 1:rows
            for c = 1:cols
                % Calculate tile boundaries
                startH = (r-1) * tileH + 1;
                endH = min(r * tileH, h);
                startW = (c-1) * tileW + 1;
                endW = min(c * tileW, w);
                
                % Extract tile
                tile = img(startH:endH, startW:endW, :);
                
                % Resize tile to target size (maintaining aspect ratio)
                [tH, tW, ~] = size(tile);
                tScaleH = tH / targetH;
                tScaleW = tW / targetW;
                tMaxScale = max(tScaleH, tScaleW);
                
                if tMaxScale > 1
                    % Tile is still larger than target, resize down
                    if tScaleH > tScaleW
                        newTileH = targetH;
                        newTileW = round(tW * targetH / tH);
                    else
                        newTileW = targetW;
                        newTileH = round(tH * targetW / tW);
                    end
                    tile = imresize(tile, [newTileH, newTileW]);
                    tH = newTileH;
                    tW = newTileW;
                end
                
                % Zero-pad tile to target size
                paddedTile = zeros(targetH, targetW, channels, 'like', img);
                
                % Center the tile
                startPadH = max(1, round((targetH - tH) / 2) + 1);
                startPadW = max(1, round((targetW - tW) / 2) + 1);
                endPadH = min(targetH, startPadH + tH - 1);
                endPadW = min(targetW, startPadW + tW - 1);
                
                % Handle edge cases
                cropTileH = min(tH, endPadH - startPadH + 1);
                cropTileW = min(tW, endPadW - startPadW + 1);
                
                paddedTile(startPadH:endPadH, startPadW:endPadW, :) = ...
                    tile(1:cropTileH, 1:cropTileW, :);
                
                result{r, c} = paddedTile;
            end
        end
    end
end