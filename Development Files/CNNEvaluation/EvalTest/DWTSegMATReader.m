function  out = SegMATReader(filename)

load(filename);

% For grayscale images, simulate RGB images by repeating the intensity
% values for all three color channels



out{1} = masks;
out{2} = boxLabel;
out{3} = boxScore;
end