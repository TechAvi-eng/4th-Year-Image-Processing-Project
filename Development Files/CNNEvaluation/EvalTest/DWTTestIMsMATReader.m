function  out = DWTTestIMsMATReader(filename)

load(filename);

% For grayscale images, simulate RGB images by repeating the intensity
% values for all three color channels

[cA,cH,cV,cD] = dwt2(im,'sym4','mode','per');
im=cat(3, cA,cH,cV,cD);
im=rescale(im); %rescale to [0,1] before inputting


out{1} = im;
end