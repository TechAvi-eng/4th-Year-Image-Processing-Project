function  out = cocoAnnotationMATReader4(filename)

load(filename);


%replace image with a 4 channel one, where each channel is a coefficient
%from a one level DWT decomposition

[cA,cH,cV,cD] = dwt2(im,'sym4','mode','per');
im=cat(3, cA,cH,cV,cD);


asd=downsample(masks, 2)



out{1} = im;
out{2} = bbox;
out{3} = label;
out{4} = masks;

end