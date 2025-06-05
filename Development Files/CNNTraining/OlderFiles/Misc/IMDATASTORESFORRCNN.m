clear
clc
close all
%load data
imds=imageDatastore("LiveCellsIms1/livecell_train_val_images/");
A=readstruct("./JSON_FORMATTING/livecell_coco_train.json");

bbox=vertcat(A.annotations.bbox);
%%
% classes=cell(size(bbox,1),1);
% classes(:,1)={"Cell"};

FileNamesIDS=[vertcat(A.images.id), vertcat(A.images.file_name)];
ImageIDS=vertcat(A.annotations.image_id);
PolyData=vertcat(A.annotations.segmentation);


%% find unique images

c=unique(ImageIDS(:,1));

 for i = 1:length(c)
   counts(i,1) = sum(ImageIDS(:,1)==c(i)); % number of times each unique value is repeated
 end

 clear c

 %% read bounding box data

clear BBOXData NewBBOXES

k=0;
for i=[1:length(counts)]

for j=[1:counts(i)]

    BBOXData(j,:)=(bbox(k+j, :));

end

NewBBOXES{i,1}=(BBOXData);
clear BBOXData

k=k+j;
end


clear i j BBOXData

NewBBOXES=table(NewBBOXES);
NewBBOXES.Properties.Variabl

blds=boxLabelDatastore(NewBBOXES);

%%
save("IMandBL_datastores.mat", "blds", "imds");