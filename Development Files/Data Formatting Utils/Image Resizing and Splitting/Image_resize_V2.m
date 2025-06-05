clc
clear all
% desired image output
% function resized=Image_resizing (desired_x_pixel,desired_y_pixel);
desired_x_pixel=50;
desired_y_pixel=50;
% Define patch size
patchHeight = desired_y_pixel; % Height of each patch
patchWidth = desired_x_pixel;  % Width of each patch

% old image input (loop for all images to process)
img=imread('tiger.jpg');
origionl_total_pixel=numel(img);
% Get the size of the image
[imageHeight, imageWidth, numChannels] = size(img);

% Create a figure to display the valid patches
figure;
patchIndex = 1;

if imageHeight>desired_y_pixel&&imageWidth>desired_x_pixel
% Breaking down larger picture and looping through the image and extract patches
for row = 1:patchHeight:imageHeight
    for col = 1:patchWidth:imageWidth
        % Check if the patch falls entirely within the image boundaries
        if row + patchHeight - 1 <= imageHeight && col + patchWidth - 1 <= imageWidth
            % Extract the patch
            patch = img(row:row + patchHeight - 1, col:col + patchWidth - 1, :);
            
            % Display the patch
            subplot(ceil(imageHeight / patchHeight), ceil(imageWidth / patchWidth), patchIndex);
            imshow(patch);
            title(sprintf('Patch %d', patchIndex));
            patchIndex = patchIndex + 1;
        else
            % Skip incomplete patches
            fprintf('Skipping patch starting at row %d, col %d (out of bounds)\n', row, col);
        end
    end
end


sgtitle('Valid Image Patches'); % Title for the entire figure

elseif imageWidth<desired_x_pixel && imageHeight<desired_y_pixel;
% resizing image
resized = imresize(img,[desired_x_pixel,desired_y_pixel]);
disp("case 1")

elseif imageWidth<=desired_x_pixel && imageHeight>=desired_y_pixel
% truncating image with rectangular crop region
x = 0;      % Starting x-coordinate
y = 0;       % Starting y-coordinate
width = desired_x_pixel;  % Width of the cropped area
height = desired_y_pixel; % Height of the cropped area
truncatedImage = imcrop(img, [x, y, width, height]);
% resizing image
resized = imresize(truncatedImage,[desired_y_pixel,desired_x_pixel]);
disp("case 2")
% displaying output
imshowpair(img,resized,"montage")
title("Input image(left), Output image(right)")
output_total_pixel=numel(resized);
output_x_pixel=size(resized,2);
output_y_pixel=size(resized,1);

elseif imageWidth>=desired_x_pixel && imageHeight<=desired_y_pixel
% truncating image with rectangular crop region
x = 0;      % Starting x-coordinate
y = 0;       % Starting y-coordinate
width = desired_x_pixel;  % Width of the cropped area
height = desired_y_pixel; % Height of the cropped area
truncatedImage = imcrop(img, [x, y, width, height]);
% resizing image
resized = imresize(truncatedImage,[desired_y_pixel,desired_x_pixel]);
disp("case 3")
% displaying output
imshowpair(img,resized,"montage")
title("Input image(left), Output image(right)")
output_total_pixel=numel(resized);
output_x_pixel=size(resized,2);
output_y_pixel=size(resized,1);

else
disp("errorrrr")

end
