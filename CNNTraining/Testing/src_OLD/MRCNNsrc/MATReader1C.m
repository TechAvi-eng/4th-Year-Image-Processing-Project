function  out = MATReader1C(filename, augmentOnOff)

load(filename);

im=single(rescale(im));

[im, masks] = resizeImageandMask(im, masks, [528, 704]);


if augmentOnOff==1
    [im, masks, label, bbox] = augmentImage(im, masks, label, bbox); %random augmentation function
end



out{1} = im;
out{2} = bbox;
out{3} = label;
out{4} = masks;

end