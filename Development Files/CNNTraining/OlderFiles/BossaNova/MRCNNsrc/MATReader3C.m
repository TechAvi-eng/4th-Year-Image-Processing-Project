function  out = MATReader3C(filename)

load(filename);

im=rescale(im);

if(size(im,3)==1)
    im = repmat(im, [1 1 3]);
end

[im, masks] = resizeImageandMask(im, masks, [528, 704]); %pad to desired size

[im, masks] = augmentImage(im, masks); %random augmentation function

out{1} = im;
out{2} = bbox;
out{3} = label;
out{4} = masks;

end