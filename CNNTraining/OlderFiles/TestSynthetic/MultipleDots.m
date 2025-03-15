clear
clc
close all


im0=zeros(520, 704);


[X Y]=meshgrid(1:704, 1:520);

cx=20;
cy=20;
rval=25;

in=1;
for cx=[50:100:650]
    for cy=[50:100:500]

R=(X-cx).^2+(Y-cy).^2;

immask=im0;
immask(R<rval^2)=1;
bbox(in,:)= [cx-rval, cy-rval, 2*rval, 2*rval ];

masks(:,:,in)=immask;
label(in,1 )=categorical({'CellA'});

in=in+1

    end
end


%%
im=sum(masks,3);
im=uint8(rescale(im)*256);
imshow(im)
showShape("rectangle", bbox, "Label", label, "LineColor",'r')

%%
clear X Y rval R in cy cx im0
