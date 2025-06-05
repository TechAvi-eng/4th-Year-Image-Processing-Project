function  out = TestMATReader(filename)

load(filename);

% For grayscale images, simulate RGB images by repeating the intensity
% values for all three color channels
masks=masks(1:2:end, 1:2:end, :);

out{1} = masks;
out{2} = label;
end