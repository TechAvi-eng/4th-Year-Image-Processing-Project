function [im, masks, labels, bbox] = CropRandom(im, masks, labels, bbox, cropSize)

    randNums = rand([1 2], 'single');
    offset=ceil( [ randNums(1)*(520-cropSize(1))+4, randNums(2)*(704-cropSize(2)) ] ); %offset crop to a random position and ensure padded area does not fall within the crop

    masks2 = masks(offset(1):offset(1)+cropSize(1)-1, offset(2):offset(2)+cropSize(2)-1, :);

    %remove masks and labels that have been cropped out (and are all zeros)
    ind = squeeze(all(masks2==0, [1,2])); 
    masks2 = masks2(:,:,~ind);
    if isempty(masks2)
        return % do not crop if crop will produce an empty result
    end
    masks=masks2;
    bbox = bbox(~ind,:);


    masks = imresize(masks, [528, 704], 'bilinear');
    labels = labels(~ind);

    

    im = im(offset(1):offset(1)+cropSize(1)-1, offset(2):offset(2)+cropSize(2)-1 );

    
    
    im = imresize(im, [528, 704], 'bilinear'); %resize image to original desired size
    


end