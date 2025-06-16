function  out = MATReaderDWT(filename)

load(filename);

im=single(rescale(im));

[im, masks] = resizeImageandMask(im, masks, [528, 704]);

% 
% if augmentOnOff==1
%     [im, masks, label, bbox] = augmentImage(im, masks, label, bbox); %random augmentation function
% end


[cA,cH,cV,cD] = dwt2(im,'sym4','mode','per');
im=cat(3, cA,cH,cV,cD);
im=rescale(im); %rescale to [0,1] before inputting


masks=masks(1:2:end, 1:2:end, :);
bbox=bbox/2;



out{1} = im;
out{2} = bbox;
out{3} = label;
out{4} = masks;

end