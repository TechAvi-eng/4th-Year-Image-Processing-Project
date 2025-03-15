clear
clc
close all


im1=zeros(520, 704);
im1(200:280, 600:680)=1;



bbox(1,:)=[600 200 80 80];
label(1)=categorical("SHSY5Y");

masks(:,:,1)=logical(im1);


im2=zeros(520, 704);
im2(400:440, 400:440)=1;
bbox(2,:)=[400 400 40 40];
label(2,1)=categorical("SHSY5Y");
masks(:,:,2)=logical(im2);



im=uint8((im1+im2)*256);




im=imnoise(im, "gaussian", 0.0002);

if(isempty(masks))
    overlayedImage = im;
else
    overlayedImage = insertObjectMask(im, masks,Color=lines(size(masks, 3)) );
end
figure, imshow(overlayedImage)

axis equal
showShape("rectangle", bbox, "Label", label, "LineColor",'r')

