clc
clear all
% desired image output
% function resized=Image_resizing (desired_x_pixel,desired_y_pixel);
desired_x_pixel=720;
desired_y_pixel=1080;

% old image input (loop for all images to process)
A=imread('feather.jpg');
origionl_total_pixel=numel(A);
origional_x_pixel=size(A,2);
origional_y_pixel=size(A,1);

if origional_x_pixel<desired_x_pixel && origional_y_pixel<desired_y_pixel||origional_x_pixel>desired_x_pixel && origional_y_pixel>desired_y_pixel;
% resizing image
resized = imresize(A,[desired_x_pixel,desired_y_pixel]);
disp("case 1")

elseif origional_x_pixel<=desired_x_pixel && origional_y_pixel>=desired_y_pixel
% truncating image with rectangular crop region
x = 0;      % Starting x-coordinate
y = 0;       % Starting y-coordinate
width = desired_x_pixel;  % Width of the cropped area
height = desired_y_pixel; % Height of the cropped area
truncatedImage = imcrop(A, [x, y, width, height]);
% resizing image
resized = imresize(truncatedImage,[desired_x_pixel,desired_y_pixel]);
disp("case 2")

elseif origional_x_pixel>=desired_x_pixel && origional_y_pixel<=desired_y_pixel
% truncating image with rectangular crop region
x = 0;      % Starting x-coordinate
y = 0;       % Starting y-coordinate
width = desired_x_pixel;  % Width of the cropped area
height = desired_y_pixel; % Height of the cropped area
truncatedImage = imcrop(A, [x, y, width, height]);
% resizing image
resized = imresize(truncatedImage,[desired_x_pixel,desired_y_pixel]);
disp("case 3")

else
disp("error")

end

% displaying output
imshowpair(A,resized,"montage")
title("Input image(left), Output image(right)")
output_total_pixel=numel(resized);
output_x_pixel=size(resized,2);
output_y_pixel=size(resized,1);
% end
