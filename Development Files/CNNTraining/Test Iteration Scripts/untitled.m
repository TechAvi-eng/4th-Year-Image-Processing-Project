close all

im = rescale(im);
noise = 12/255*randn([520 704],"double");

A = im+noise;

%A = DWT_Denoise(A);

imshow(A)

