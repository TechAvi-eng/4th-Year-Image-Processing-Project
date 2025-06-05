

im=rescale(im);
[im, masks] = resizeImageandMask(im, masks, [528, 704]);


[mask,labels,scores,boxes] = segmentObjects(net,im,Threshold=0.1,NumStrongestRegions=Inf, SelectStrongest=true, MinSize=[1 1],MaxSize=[80 80] );

%%
overlayedImage = insertObjectMask(im, masks,Color=[0.9290 0.6940 0.1250] );
overlayedImage = insertObjectMask(overlayedImage, mask,Color=[0.3010 0.7450 0.9330, 0.3] );

figure, imshow(overlayedImage)

