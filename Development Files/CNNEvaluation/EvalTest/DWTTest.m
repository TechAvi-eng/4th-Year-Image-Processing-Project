


load("SingleDS/label_A172_Phase_A7_1_00d00h00m_2.tif.mat", "im")
im=rescale(im);
im=repmat(im ,[1 1 1]); 

[cA,cH,cV,cD] = dwt2(im,'sym4','mode','per');
im=cat(3, cA,cH,cV,cD);
im=rescale(im); %rescale to [0,1] before inputting

%%

[masks,labels,scores,boxes] = segmentObjects(net,im(1:2:end, 1:2:end, :) ,Threshold=0.2,NumStrongestRegions=inf, SelectStrongest=true, MinSize=[1 1],MaxSize=[80 80] );
if(isempty(masks))
    overlayedImage = im(:,:,1);
else
    overlayedImage = insertObjectMask(im(1:2:end,1:2:end,1), masks,Color=lines(size(masks, 3)) );
end

figure, imshow(overlayedImage)