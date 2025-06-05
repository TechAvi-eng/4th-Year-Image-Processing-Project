function [masks labels scores boxes] = segmentCells(net, Image, Options)
% SEGMENTCELLS De-noises image via DWT-thresholding and uses a Mask R-CNN network to segment cells in an image
    arguments %checks and default values
        net MRCNN %classification network input
        Image = []; %image to be classified
        Options.Denoise (1,1) logical =1; %on-off option for denoising process
        Options.Wavelet char = 'db5' %wavelet to use in denoising
        Options.Level (1,1) {mustBeInteger, mustBeReal} = 4 %levels of decomposition in de-noising
        Options.DWTThreshold {mustBeGreaterThanOrEqual(Options.DWTThreshold, 0), mustBeLessThan(Options.DWTThreshold, 1), mustBeReal(Options.DWTThreshold)}= 0.02; %threshold for decomposition
        Options.SegmentThreshold (1,1) {mustBeGreaterThanOrEqual(Options.SegmentThreshold, 0), mustBeLessThan(Options.SegmentThreshold, 1), mustBeReal(Options.SegmentThreshold)}= 0.5; %segmentation confidence threshold for MRCNN results
        Options.NumstrongestRegions (1,1) = 5000; %number of regions to take forward from RPN
        Options.SelectStrongest logical = 1; 
        Options.MinSize (1,2) = [2 2]; %minimum segmenteed object size
        Options.MaxSize (1,2) = [200 200]; %minimum segmenteed object size
        Options.ShowMasks (1,1) logical = 0; %plot showing masks
        Options.ShowScores (1,1) logical = 0; %add scores to plot
        Options.ExecutionEnvironment char = 'cpu'; %execution environment, cpu or GPU
    end

if Options.Denoise==1 %apply denoising
    Image = DWT_Denoise(Image, "Level",Options.Level,"Threshold",Options.DWTThreshold,"Wavelet",Options.Wavelet);
end

Image = rescale(Image); %scale [0,1] to pass to Mask R CNN Network
[masks,labels,scores,boxes] = segmentObjects(net,Image,Threshold=Options.SegmentThreshold,NumStrongestRegions=Options.NumstrongestRegions, SelectStrongest=Options.SelectStrongest, MinSize=Options.MinSize,MaxSize=Options.MaxSize, ExecutionEnvironment=Options.ExecutionEnvironment);

if Options.ShowMasks==1 %display masks
    if(isempty(masks))
        overlayedImage = Image(:,:,1);
    else
        overlayedImage = insertObjectMask(Image(:,:,1), masks,Color=lines(size(masks, 3)) );
    end
    
    figure, imshow(overlayedImage)
    if Options.ShowScores==1 %display bounding boxes and confidence scores [0,1]
        showShape("rectangle", gather(boxes), "Label", scores, "LineColor",'r');
    end


end