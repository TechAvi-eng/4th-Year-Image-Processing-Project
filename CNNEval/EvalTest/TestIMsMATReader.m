function  out = TestIMsMATReader(filename)

load(filename);

im = rescale(im);

%im = DWT_Denoise(im);

sz = [520 704];
paddedH = 528;
paddedW = 704;

c1 = ceil((paddedH - sz(1))/2)+1:ceil((paddedH + sz(1))/2 );
c2 = ceil((paddedW - sz(2))/2 )+1:ceil((paddedW + sz(2))/2 );

imageOut = zeros(paddedH,paddedW,size(im, 3), 'like', im);
imageOut(c1, c2, :) = im; %pad equally from all sides

im = imageOut;
% For grayscale images, simulate RGB images by repeating the intensity
% values for all three color channels


out{1} = im;
end