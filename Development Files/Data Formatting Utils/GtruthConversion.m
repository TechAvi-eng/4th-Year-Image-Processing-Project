clear
clc
close all

A=readstruct("livecell_coco_train.json");
load("LabelDefs.mat");

%% format data

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

%% find unique images
c=unique(ImageIDS(:,1));

 for i = 1:length(c)
   counts(i,1) = sum(ImageIDS(:,1)==c(i)); % number of times each unique value is repeated
 end

 clear c


%% set labels into table
clear CellData NewLabelData

k=0;
for i=[1:length(counts)]

for j=[1:counts(i)]

     Temp=single([PolyData{k+j,1}(1:2:end); PolyData{k+j,1}(2:2:end)]');
    CellData{j,1}=(Temp);

end

NewLabelData{i,1}=(CellData);
clear CellData

k=k+j;
end


clear Temp i j


NewLabelData=cell2table(NewLabelData);
NewLabelData.Properties.VariableNames="Cell1";

%% set image paths (note that they the livecell images are stored as is in a folder called LiveCellsIms1 in the working directory, this is not uploaded due to data limiations

imagepath="LiveCellsIms1/livecell_train_val_images/";
dataSource=[imagepath+""];


sources=([imagepath+(FileNamesIDS(:,2)) ]);

NewSources=groundTruthDataSource(sources);


%%
newGTruth = groundTruth(NewSources, labeldefs, NewLabelData);