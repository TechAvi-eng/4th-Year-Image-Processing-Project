clc
close all
clear

im=imread("4_00003.png");
imgs=mean(im,3);
imgs=rescale(imgs);
imshow(imgs)


%%

[c,s]=wavedec2(imgs,5,'haar');

[H1,V1,D1] = detcoef2('all',c,s,2);
A1 = appcoef2(c,s,'haar',1);


V1img = wcodemat(V1,255,'mat',1);
H1img = wcodemat(H1,255,'mat',1);
D1img = wcodemat(D1,255,'mat',1);
A1img = wcodemat(A1,255,'mat',1);

tiledlayout(2,2,"TileSpacing","tight")
nexttile

imagesc(A1img)
colormap pink(255)
title('Approximation', Interpreter='latex')
colormap("gray")
set(gca,'XTick',[])
set(gca,'YTick',[])

nexttile
imagesc(H1img)
title('Horizontal Detail', Interpreter='latex')
colormap("gray")
set(gca,'XTick',[])
set(gca,'YTick',[])

nexttile
imagesc(V1img)
xlabel('Vertical Detail', Interpreter='latex')
colormap("gray")
set(gca,'XTick',[])
set(gca,'YTick',[])

nexttile
imagesc(D1img)
xlabel('Diagonal Detail', Interpreter='latex')
colormap("gray")
set(gca,'XTick',[])
set(gca,'YTick',[])




%%
clear NewImg
clc

NewImg(:,:,1)=H1img ;
NewImg(:,:,2)=V1img ;
NewImg(:,:,3)=D1img ;

%%
imshow(rescale(NewImg))