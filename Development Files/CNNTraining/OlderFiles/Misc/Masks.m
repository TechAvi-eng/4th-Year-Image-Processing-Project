clear
clc
close all

%A=readstruct("./JSON_FORMATTING/livecell_coco_train.json");

A=readstruct("./JSON_FORMATTING/0_train2percent.json");

FileNamesIDS=[vertcat(A.images.id), vertcat(A.images.file_name)];




ImageIDS=vertcat(A.annotations.image_id);


PolyData=vertcat(A.annotations.segmentation);
list=[];
for j=[1:length(PolyData)]
    if class(PolyData{j,1})=="struct"
    list=[list j];
    end
end

PolyData(list)=[];
ImageIDS(list)=[];



%% turn polygon labels for cells into binary masks
nocells=length(PolyData);

[h w]=size(imread("R_CNN_TEST1/A172_Phase_C7_1_00d00h00m_1.tif"));
denseMasks = false([h,w,nocells]);
%%
for i=1:nocells
    polygon=cell2mat(PolyData(i,1));

    denseMasks(:,:,i)=poly2mask( polygon(1,1:2:end), polygon(1,2:2:end),h, w);

end

% %% check masks for an image
%imshow(denseMasks(:,:,2100));