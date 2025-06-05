clear all
close all
clc

% Define folder containing test images
folder_path = 'test_images';
imageFiles = dir(fullfile(folder_path, '*.png'));

% Initialisation of parameters
imageNumber = 1;

% Get the full image path
imagePath = fullfile(folder_path, imageFiles(imageNumber).name);

% Importing image
im=imread(imagePath);

% Coverts image to grayscale by taking mean of the 3 colour channels
img = mean(im,3);
img = rescale(img);

% Generate denoised images
level_3 = DWT_Denoise(img, "Threshold", 0.1, "Level", 3, "Wavelet", 'db5');
level_4 = DWT_Denoise(img, "Threshold", 0.1, "Level", 4, "Wavelet", 'db5');
level_5 = DWT_Denoise(img, "Threshold", 0.1, "Level", 5, "Wavelet", 'db5');

% Apply global settings for consistency
set(groot, 'DefaultTextInterpreter', 'latex');
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
set(groot, 'DefaultLegendInterpreter', 'latex');
set(groot, 'DefaultAxesFontName', 'CMU Serif');
set(groot, 'DefaultTextFontName', 'CMU Serif');

% Create figure for subplots
figure;
t = tiledlayout(1,3,'Padding','None','TileSpacing','compact');

% Level 3 image

nexttile(1)
imshow(level_3);
title('Level 3 DWT Denoised', 'Interpreter', 'latex', 'FontName', 'CMU Serif');

% Level 4 image
nexttile(2)
imshow(level_4);
title('Level 4 DWT Denoised', 'Interpreter', 'latex', 'FontName', 'CMU Serif');

% Level 5 image
nexttile(3)
imshow(level_5);
title('Level 5 DWT Denoised', 'Interpreter', 'latex', 'FontName', 'CMU Serif');

