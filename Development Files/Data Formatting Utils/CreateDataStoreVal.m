clear
clc
close all
%load and read data and appropriate columns
A=readstruct("livecell_coco_val.json"); %file containing json data
 bboxDat=vertcat(A.annotations.bbox);
FileNamesIDS=[vertcat(A.images.id), vertcat(A.images.file_name)];
ImageIDS=vertcat(A.annotations.image_id);
PolyData=vertcat(A.annotations.segmentation);
%% get rid of other data formats; clean comments etc

list=[];
for j=[1:length(PolyData)]
    if class(PolyData{j,1})=="struct"
    list=[list j];
    end
end

PolyData(list)=[];
ImageIDS(list)=[];
clear list j

%% find unique images
c=unique(ImageIDS(:,1));

 for i = 1:length(c)
   counts(i,1) = sum(ImageIDS(:,1)==c(i)); % number of times each unique value is repeated
 end

 clear c

 %%

[h w]=size(imread("./LiveCellsIms1/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif"));

imagelocs=[0; cumsum(counts)]; %starting position offset for each image in the list of data 

%%

for i=[1:length(counts)] %for the number of images


    im = imread( ["./LiveCellsIms1/livecell_train_val_images/"+FileNamesIDS{i, 2   } ]   ); %change with image path as needed
        % if size(im, 3)==1
        %     im=repmat(im, [1 1 3]); %turn image into pseudo RGB (all three channels the same) if in B&W, as this is required by the NN
        % end


        masks = false([h,w,counts(i)]); %where counts(i) is the number of segments in that image
        for j=[1:counts(i)] %can also use parfor here if needed but sometimes results in errors
            bbox(j,:)=(bboxDat(imagelocs(i)+j, :)); %starting position offset (imagelocs...)+the index within the current image
            polygon=cell2mat(PolyData(imagelocs(i)+j,1));
            masks(:,:,j)=poly2mask( polygon(1,1:2:end), polygon(1,2:2:end),h, w); %turn polygon data into binary masks
        end
        label(1:counts(i),1 )="CellA"; %all cells are annotated as CellA at the moment
        label=categorical(label);
        %imshow(denseMasks(:,:,2))

        save(["ValDSFs/label_"+FileNamesIDS{i, 2}+".mat"], "im", "bbox","label", "masks") %write data for current image to a .mat file
        clear bbox polygon masks label j %clear data so now ones can be written

        disp(["Image: "+i+" Percent: "+i/length(imagelocs)*100+"%"]) %print progress

end









%%  test image
% imshow(im)
% imOverlay = insertObjectMask(im,masks,Color=lines(size(masks, 3)));
% imshow(imOverlay);
