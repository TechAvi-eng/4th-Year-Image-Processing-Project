function [im, masks, labels, bbox] = augmentImage(im, masks, labels, bbox)

randNums = rand([1 8],"single"); %generate random numbers for augmentation, more efficient to do so in one step

if randNums(1)<0.2 %horizontal flip with 20% probability
    im = im(end:-1:1, :);
    masks = masks(end:-1:1, :, :);
end

if randNums(2)<0.2 %vertical flip with 20% probability
    im = im(:, end:-1:1);
    masks = masks(:, end:-1:1, :);
end

if randNums(3)<0.1 %artificially increase contrast with a sigmoid-like function
    im = ( -cos( pi * im ) + 1 ) / 2;
    if randNums(3)<0.01
    %in rare cases, increase it to a really high level of contrast
    im = ( -cos( pi * im ) + 1 ) / 2;
    end
end

if randNums(3)>0.95 %artificially decrease contrast (inverse of previous function), note it cannot happen at the same time as an increase
    im = 1 - acos( 2*(im-0.5) )/pi;
end

% crop to smaller sizes randomly, change masks and labels to remove empty elements
if randNums(6)<0.025
    [im, masks, labels, bbox] = CropRandom(im, masks, labels, bbox, [264, 352]);
end

if randNums(4)<0.25 %add Gaussian white noise with 40% probability
    im = imnoise(im, "gaussian", 0, 0.0025*randNums(5)); %mean 0, variance chosen randomly
end


end
