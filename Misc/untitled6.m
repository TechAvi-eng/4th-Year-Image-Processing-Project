
im= rescale(im);

im2=(im+randn(size(im))*20/255 );

imshow(im2);

im3=DWT_Denoise(im2);
imshow(im3)