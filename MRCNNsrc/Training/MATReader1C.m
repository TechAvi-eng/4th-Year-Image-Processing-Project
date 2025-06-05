function  out = MATReader1C(filename, augmentOnOff)

load(filename);

im=rescale(im);

[im, masks] = resizeImageandMask(im, masks, [528, 704]);
bbox(:,2) = min(bbox(:,2)+4, 528); %will imlpement into resize function later, see bug report

if augmentOnOff==1
    [im, masks, label, bbox] = augmentImage(im, masks, label, bbox); %random augmentation function
end



out{1} = im;
out{2} = bbox;
out{3} = label;
out{4} = masks;

end