function  out = MATReader1C(filename)

load(filename);

% For grayscale images, simulate RGB images by repeating the intensity
% values for all three color channels

out{1} = im;
out{2} = bbox;
out{3} = label;
out{4} = masks;

end