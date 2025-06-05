clear all
close all
clc

% MAKE SURE THE FOLDER CONTAINING IMAGES IS ADDED TO PATH AND YOU ARE IN
% THE SAME FOLDER HIERACHY AS THE IMAGE FOLDER

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

% Initialise Parameters (optional, if not specified then remove from
% function call)
wavelet = 'db5';
n = 4; % Bug means current max n = 6
threshold_in = [0.2 0.2 0.25 0.25]; % Threshold (decimal). Alternatively, threshold_in could be a single scalar value.

% Applies the DWT thresholding function and returns the denoised image
denoised_img = Coefficients(img, wavelet, n, threshold_in);
