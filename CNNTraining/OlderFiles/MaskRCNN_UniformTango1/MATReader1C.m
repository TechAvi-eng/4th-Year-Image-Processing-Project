function  out = cocoAnnotationMATReader(filename)

load(filename);

% For grayscale images, simulate RGB images by repeating the intensity
% values for all three color channels
%im=imnoise(im, "gaussian", 0.0002);

im=uint8(rescale(im,0, 255));

out{1} = im;
out{2} = bbox;
out{3} = label;
out{4} = masks;

end